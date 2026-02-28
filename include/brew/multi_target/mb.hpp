#pragma once

#include "brew/multi_target/rfs_base.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/bernoulli.hpp"
#include "brew/models/trajectory_base_model.hpp"
#include "brew/filters/filter.hpp"
#include "brew/clustering/dbscan.hpp"
#include "brew/assignment/hungarian.hpp"

#include <Eigen/Dense>
#include <memory>
#include <type_traits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>


namespace brew::multi_target {

/// Multi-Bernoulli (MB) filter.
/// Single-hypothesis unlabeled filter using Hungarian (single best) assignment.
/// No persistent track identity — Bernoullis are a flat set without labels.
/// Template parameter T is the single distribution type (e.g., Gaussian, GGIW).
template <typename T>
class MB : public RFSBase {
public:
    MB() = default;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<MB<T>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        for (const auto& b : bernoullis_) {
            c->bernoullis_.push_back(b->clone());
        }
        for (const auto& b : birth_bernoullis_) {
            c->birth_bernoullis_.push_back(b->clone());
        }
        c->prune_threshold_bern_ = prune_threshold_bern_;
        c->extract_threshold_ = extract_threshold_;
        c->gate_threshold_ = gate_threshold_;
        c->next_track_id_ = next_track_id_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        return c;
    }

    // ---- Configuration ----

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
    }

    void set_birth_bernoullis(std::vector<std::unique_ptr<models::Bernoulli<T>>> birth) {
        birth_bernoullis_ = std::move(birth);
    }

    void set_birth_model(std::unique_ptr<models::Mixture<T>> birth_mix) {
        birth_bernoullis_.clear();
        if (!birth_mix) return;
        if (!birth_mix->empty()) {
            is_extended_ = birth_mix->component(0).is_extended();
            if (is_extended_ && !cluster_obj_) {
                cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
        for (std::size_t i = 0; i < birth_mix->size(); ++i) {
            birth_bernoullis_.push_back(std::make_unique<models::Bernoulli<T>>(
                birth_mix->weight(i),
                birth_mix->component(i).clone_typed()));
        }
    }

    void set_prune_threshold_bernoulli(double t) { prune_threshold_bern_ = t; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::DBSCAN> obj) { cluster_obj_ = std::move(obj); }

    // ---- Accessors ----

    [[nodiscard]] const std::vector<std::unique_ptr<models::Mixture<T>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    // ---- RFS interface ----

    void predict(int /*timestep*/, double dt) override {
        if (!filter_) return;

        // Propagate existing Bernoullis
        for (auto& bern : bernoullis_) {
            if (!bern->has_distribution()) continue;
            bern->set_existence_probability(bern->existence_probability() * prob_survive_);
            auto predicted = filter_->predict(dt, bern->distribution());
            bern->set_distribution(predicted.clone_typed());
        }

        // Add birth Bernoullis with new track IDs
        for (const auto& bb : birth_bernoullis_) {
            auto new_bern = bb->clone();
            new_bern->set_id(next_track_id_++);
            if (new_bern->has_distribution()) {
                increment_init_idx(new_bern->distribution());
            }
            bernoullis_.push_back(std::move(new_bern));
        }

        // Increment init_idx on birth model for trajectory types
        for (const auto& bb : birth_bernoullis_) {
            if (bb->has_distribution()) {
                increment_init_idx(const_cast<T&>(bb->distribution()));
            }
        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!filter_) return;

        // Build measurement groups
        std::vector<Eigen::MatrixXd> meas_groups;
        if (is_extended_ && cluster_obj_) {
            meas_groups = cluster_obj_->cluster(measurements);
        } else {
            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }

        const int m = static_cast<int>(meas_groups.size());
        if (m == 0) {
            // No measurements: all Bernoullis get missed-detection update
            correct_no_measurements();
            return;
        }

        const int n_bern = static_cast<int>(bernoullis_.size());
        if (n_bern == 0) return;

        const double kappa_base = clutter_rate_ * clutter_density_;

        std::vector<double> kappa_vec(m);
        for (int j = 0; j < m; ++j) {
            const int W = static_cast<int>(meas_groups[j].cols());
            kappa_vec[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
        }

        // Build cost matrix: n_bern rows x (m + n_bern) cols
        const int n_cols = m + n_bern;
        Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(n_bern, n_cols,
            std::numeric_limits<double>::infinity());

        // Cache corrected distributions and likelihoods
        struct CorrectionCache {
            std::unique_ptr<T> dist;
            double likelihood = 0.0;
        };
        std::vector<std::vector<CorrectionCache>> cache(n_bern);
        for (int i = 0; i < n_bern; ++i) cache[i].resize(m);

        for (int i = 0; i < n_bern; ++i) {
            const auto& bern = *bernoullis_[i];
            double r = bern.existence_probability();
            if (!bern.has_distribution() || r <= 0.0) continue;

            for (int j = 0; j < m; ++j) {
                const Eigen::MatrixXd& meas = meas_groups[j];
                Eigen::VectorXd z_gate = (is_extended_ && meas.cols() > 1)
                    ? Eigen::VectorXd(meas.rowwise().mean())
                    : Eigen::VectorXd(meas.col(0));

                double gate_val = filter_->gate(z_gate, bern.distribution());
                if (gate_val < gate_threshold_) {
                    Eigen::VectorXd meas_flat;
                    if (meas.cols() > 1) {
                        meas_flat.resize(meas.size());
                        for (int c = 0; c < meas.cols(); ++c)
                            meas_flat.segment(c * meas.rows(), meas.rows()) = meas.col(c);
                    } else {
                        meas_flat = meas.col(0);
                    }

                    auto [dist, qz] = filter_->correct(meas_flat, bern.distribution());
                    cache[i][j].dist = dist.clone_typed();
                    cache[i][j].likelihood = qz;

                    double det_likelihood = prob_detection_ * r * qz;
                    if (det_likelihood > 0.0 && kappa_vec[j] > 0.0) {
                        cost(i, j) = -std::log(det_likelihood / kappa_vec[j]);
                    }
                }
            }

            // Missed detection cost (diagonal column m+i)
            double miss_factor = 1.0 - prob_detection_ * r;
            if (miss_factor > 0.0) {
                cost(i, m + i) = -std::log(miss_factor);
            } else {
                cost(i, m + i) = 0.0;
            }
        }

        // Solve with Hungarian (single best assignment)
        auto sol = assignment::hungarian(cost);

        // Determine which Bernoullis are detected vs missed
        std::vector<int> bern_to_meas(n_bern, -1);
        for (const auto& [row, col] : sol.assignments) {
            if (row < n_bern && col < m) {
                bern_to_meas[row] = col;
            }
        }

        // Update Bernoullis in-place
        std::vector<std::unique_ptr<models::Bernoulli<T>>> updated;
        for (int i = 0; i < n_bern; ++i) {
            const auto& orig = *bernoullis_[i];
            double r = orig.existence_probability();

            if (bern_to_meas[i] >= 0) {
                int j = bern_to_meas[i];
                updated.push_back(std::make_unique<models::Bernoulli<T>>(
                    1.0, cache[i][j].dist->clone_typed(), orig.id()));
            } else {
                double r_new = r * (1.0 - prob_detection_)
                               / (1.0 - r * prob_detection_);
                updated.push_back(std::make_unique<models::Bernoulli<T>>(
                    r_new, orig.distribution().clone_typed(), orig.id()));
            }
        }

        bernoullis_ = std::move(updated);
    }

    void cleanup() override {
        // Prune low-existence Bernoullis
        std::vector<std::unique_ptr<models::Bernoulli<T>>> kept;
        for (auto& b : bernoullis_) {
            if (b->existence_probability() >= prune_threshold_bern_) {
                kept.push_back(std::move(b));
            }
        }
        bernoullis_ = std::move(kept);

        // Extract
        extracted_mixtures_.push_back(extract());
    }

    [[nodiscard]] std::unique_ptr<models::Mixture<T>> extract() const {
        auto result = std::make_unique<models::Mixture<T>>();
        for (const auto& bern : bernoullis_) {
            if (bern->existence_probability() >= extract_threshold_ && bern->has_distribution()) {
                result->add_component(
                    bern->distribution().clone_typed(),
                    bern->existence_probability());
            }
        }
        return result;
    }

private:
    void correct_no_measurements() {
        for (auto& bern : bernoullis_) {
            double r = bern->existence_probability();
            double r_new = r * (1.0 - prob_detection_)
                           / (1.0 - r * prob_detection_);
            bern->set_existence_probability(r_new);
        }
    }

    template <typename U>
<<<<<<< Updated upstream
    static void increment_init_idx(U& /*dist*/) {}

    static void increment_init_idx(models::TrajectoryBaseModel& dist) {
        dist.init_idx += 1;
=======
    static void increment_init_idx(U& dist) {
        if constexpr (std::is_base_of_v<distributions::TrajectoryBaseModel, U>) {
            dist.init_idx += 1;
        }
>>>>>>> Stashed changes
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    std::vector<std::unique_ptr<models::Bernoulli<T>>> bernoullis_;
    std::vector<std::unique_ptr<models::Bernoulli<T>>> birth_bernoullis_;

    double prune_threshold_bern_ = 1e-3;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    int next_track_id_ = 0;
    bool is_extended_ = false;
    std::shared_ptr<clustering::DBSCAN> cluster_obj_;
    std::vector<std::unique_ptr<models::Mixture<T>>> extracted_mixtures_;
};

} // namespace brew::multi_target
