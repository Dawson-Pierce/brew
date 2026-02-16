#pragma once

#include "brew/multi_target/rfs_base.hpp"
#include "brew/distributions/mixture.hpp"
#include "brew/distributions/bernoulli.hpp"
#include "brew/distributions/trajectory_base_model.hpp"
#include "brew/filters/filter.hpp"
#include "brew/clustering/dbscan.hpp"
#include "brew/assignment/murty.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace brew::multi_target {

/// Labeled Multi-Bernoulli (LMB) filter.
/// Uses delta-GLMB update internally, then marginalizes back to a single
/// labeled Bernoulli set. Template parameter T is the distribution type.
template <typename T>
class LMB : public RFSBase {
public:
    LMB() = default;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<LMB<T>>();
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
        c->k_best_ = k_best_;
        c->next_track_id_ = next_track_id_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        c->track_histories_ = track_histories_;
        return c;
    }

    // ---- Configuration ----

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
    }

    void set_birth_bernoullis(std::vector<std::unique_ptr<distributions::Bernoulli<T>>> birth) {
        birth_bernoullis_ = std::move(birth);
    }

    void set_birth_model(std::unique_ptr<distributions::Mixture<T>> birth_mix) {
        birth_bernoullis_.clear();
        if (!birth_mix) return;
        if (!birth_mix->empty()) {
            is_extended_ = birth_mix->component(0).is_extended();
        }
        for (std::size_t i = 0; i < birth_mix->size(); ++i) {
            birth_bernoullis_.push_back(std::make_unique<distributions::Bernoulli<T>>(
                birth_mix->weight(i),
                birth_mix->component(i).clone_typed()));
        }
    }

    void set_prune_threshold_bernoulli(double t) { prune_threshold_bern_ = t; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_k_best(int k) { k_best_ = k; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::DBSCAN> obj) { cluster_obj_ = std::move(obj); }

    // ---- Accessors ----

    [[nodiscard]] const std::vector<std::unique_ptr<distributions::Mixture<T>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    [[nodiscard]] const std::map<int, std::vector<Eigen::VectorXd>>& track_histories() const {
        return track_histories_;
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

            double miss_factor = 1.0 - prob_detection_ * r;
            if (miss_factor > 0.0) {
                cost(i, m + i) = -std::log(miss_factor);
            }
            // else: leave at infinity — missing a certainly-detected target is forbidden
        }

        // Solve K-best assignments (delta-GLMB update)
        auto solutions = assignment::murty(cost, k_best_);

        if (solutions.empty()) {
            correct_no_measurements();
            return;
        }

        // Build hypotheses: each solution gives a weighted set of updated Bernoullis
        struct Hypothesis {
            double log_weight;
            std::vector<std::unique_ptr<distributions::Bernoulli<T>>> bernoullis;
        };

        std::vector<Hypothesis> hypotheses;
        for (const auto& sol : solutions) {
            Hypothesis hyp;
            hyp.log_weight = -sol.total_cost;

            std::vector<int> bern_to_meas(n_bern, -1);
            for (const auto& [row, col] : sol.assignments) {
                if (row < n_bern && col < m) {
                    bern_to_meas[row] = col;
                }
            }

            for (int i = 0; i < n_bern; ++i) {
                const auto& orig = *bernoullis_[i];
                double r = orig.existence_probability();

                if (bern_to_meas[i] >= 0) {
                    int j = bern_to_meas[i];
                    hyp.bernoullis.push_back(std::make_unique<distributions::Bernoulli<T>>(
                        1.0, cache[i][j].dist->clone_typed(), orig.id()));
                } else {
                    double r_new = r * (1.0 - prob_detection_)
                                   / (1.0 - r * prob_detection_);
                    hyp.bernoullis.push_back(std::make_unique<distributions::Bernoulli<T>>(
                        r_new, orig.distribution().clone_typed(), orig.id()));
                }
            }

            hypotheses.push_back(std::move(hyp));
        }

        // Normalize hypothesis weights
        double max_lw = hypotheses[0].log_weight;
        for (const auto& h : hypotheses) {
            max_lw = std::max(max_lw, h.log_weight);
        }
        double sum_exp = 0.0;
        for (auto& h : hypotheses) {
            sum_exp += std::exp(h.log_weight - max_lw);
        }
        double log_norm = max_lw + std::log(sum_exp);
        std::vector<double> weights(hypotheses.size());
        for (std::size_t h = 0; h < hypotheses.size(); ++h) {
            weights[h] = std::exp(hypotheses[h].log_weight - log_norm);
        }

        // Marginalize across hypotheses to produce single LMB
        // For each track ID: r = sum_h w_h * r_h, dist from highest-weight hypothesis
        std::map<int, double> marginal_r;
        std::map<int, std::unique_ptr<T>> marginal_dist;
        std::map<int, int> marginal_id;

        // Sort hypotheses by weight (descending) so first encounter has best dist
        std::vector<std::size_t> hyp_order(hypotheses.size());
        std::iota(hyp_order.begin(), hyp_order.end(), 0);
        std::sort(hyp_order.begin(), hyp_order.end(),
            [&weights](std::size_t a, std::size_t b) { return weights[a] > weights[b]; });

        for (auto h_idx : hyp_order) {
            const auto& hyp = hypotheses[h_idx];
            for (std::size_t i = 0; i < hyp.bernoullis.size(); ++i) {
                int tid = hyp.bernoullis[i]->id();
                double r_h = hyp.bernoullis[i]->existence_probability();

                marginal_r[tid] += weights[h_idx] * r_h;

                // Use distribution from highest-weight hypothesis (first encounter)
                if (marginal_dist.find(tid) == marginal_dist.end()) {
                    if (hyp.bernoullis[i]->has_distribution()) {
                        marginal_dist[tid] = hyp.bernoullis[i]->distribution().clone_typed();
                        marginal_id[tid] = tid;
                    }
                }
            }
        }

        // Replace bernoullis_ with marginalized set
        bernoullis_.clear();
        for (auto& [tid, r] : marginal_r) {
            if (marginal_dist.find(tid) != marginal_dist.end()) {
                bernoullis_.push_back(std::make_unique<distributions::Bernoulli<T>>(
                    r, std::move(marginal_dist[tid]), tid));
            }
        }
    }

    void cleanup() override {
        // Prune low-existence Bernoullis
        std::vector<std::unique_ptr<distributions::Bernoulli<T>>> kept;
        for (auto& b : bernoullis_) {
            if (b->existence_probability() >= prune_threshold_bern_) {
                kept.push_back(std::move(b));
            }
        }
        bernoullis_ = std::move(kept);

        // Record track states
        record_track_states();

        // Extract
        extracted_mixtures_.push_back(extract());
    }

    [[nodiscard]] std::unique_ptr<distributions::Mixture<T>> extract() const {
        auto result = std::make_unique<distributions::Mixture<T>>();
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
    static void increment_init_idx(U& /*dist*/) {}

    static void increment_init_idx(distributions::TrajectoryBaseModel& dist) {
        dist.init_idx += 1;
    }

    template <typename U, typename = void>
    struct has_get_last_state_ : std::false_type {};

    template <typename U>
    struct has_get_last_state_<U,
        std::void_t<decltype(std::declval<const U&>().get_last_state())>
    > : std::true_type {};

    template <typename U>
    static Eigen::VectorXd get_track_state(const U& dist) {
        if constexpr (has_get_last_state_<U>::value) {
            return dist.get_last_state();
        } else {
            return dist.mean();
        }
    }

    void record_track_states() {
        for (const auto& bern : bernoullis_) {
            if (bern->existence_probability() >= extract_threshold_ && bern->has_distribution()) {
                track_histories_[bern->id()].push_back(get_track_state(bern->distribution()));
            }
        }
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    std::vector<std::unique_ptr<distributions::Bernoulli<T>>> bernoullis_;
    std::vector<std::unique_ptr<distributions::Bernoulli<T>>> birth_bernoullis_;

    double prune_threshold_bern_ = 1e-3;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    int k_best_ = 5;
    int next_track_id_ = 0;
    bool is_extended_ = false;
    std::shared_ptr<clustering::DBSCAN> cluster_obj_;
    std::vector<std::unique_ptr<distributions::Mixture<T>>> extracted_mixtures_;
    std::map<int, std::vector<Eigen::VectorXd>> track_histories_;
};

} // namespace brew::multi_target
