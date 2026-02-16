#pragma once

#include "brew/multi_target/rfs_base.hpp"
#include "brew/multi_target/mbm.hpp"
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
#include <numeric>
#include <limits>
#include <type_traits>

namespace brew::multi_target {

/// Generalized Labeled Multi-Bernoulli (delta-GLMB) filter.
/// Multi-hypothesis labeled filter with explicit cardinality estimation.
/// Structurally similar to MBM but with label-set-aware component management
/// and cardinality distribution computation.
/// Template parameter T is the single distribution type (e.g., Gaussian, GGIW).
template <typename T>
class GLMB : public RFSBase {
public:
    GLMB() = default;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<GLMB<T>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        for (const auto& b : bernoullis_) {
            c->bernoullis_.push_back(b->clone());
        }
        c->global_hypotheses_ = global_hypotheses_;
        for (const auto& b : birth_bernoullis_) {
            c->birth_bernoullis_.push_back(b->clone());
        }
        c->prune_threshold_hyp_ = prune_threshold_hyp_;
        c->prune_threshold_bern_ = prune_threshold_bern_;
        c->max_hypotheses_ = max_hypotheses_;
        c->extract_threshold_ = extract_threshold_;
        c->gate_threshold_ = gate_threshold_;
        c->k_best_ = k_best_;
        c->next_track_id_ = next_track_id_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        c->track_histories_ = track_histories_;
        c->cardinality_pmf_ = cardinality_pmf_;
        c->estimated_cardinality_ = estimated_cardinality_;
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
            if (is_extended_ && !cluster_obj_) {
                cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
        for (std::size_t i = 0; i < birth_mix->size(); ++i) {
            birth_bernoullis_.push_back(std::make_unique<distributions::Bernoulli<T>>(
                birth_mix->weight(i),
                birth_mix->component(i).clone_typed()));
        }
    }

    void set_prune_threshold_hypothesis(double t) { prune_threshold_hyp_ = t; }
    void set_prune_threshold_bernoulli(double t) { prune_threshold_bern_ = t; }
    void set_max_hypotheses(int n) { max_hypotheses_ = n; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_k_best(int k) { k_best_ = k; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::DBSCAN> obj) { cluster_obj_ = std::move(obj); }

    // ---- Accessors ----

    [[nodiscard]] const std::vector<GlobalHypothesis>& global_hypotheses() const {
        return global_hypotheses_;
    }

    [[nodiscard]] const std::vector<std::unique_ptr<distributions::Mixture<T>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    [[nodiscard]] const std::map<int, std::vector<Eigen::VectorXd>>& track_histories() const {
        return track_histories_;
    }

    /// Estimated cardinality (mean of cardinality PMF).
    [[nodiscard]] double estimated_cardinality() const {
        return estimated_cardinality_;
    }

    /// Cardinality probability mass function.
    [[nodiscard]] const Eigen::VectorXd& cardinality() const {
        return cardinality_pmf_;
    }

    // ---- RFS interface ----

    void predict(int /*timestep*/, double dt) override {
        if (!filter_) return;

        // Propagate existing Bernoulli components
        for (auto& bern : bernoullis_) {
            if (!bern->has_distribution()) continue;
            bern->set_existence_probability(bern->existence_probability() * prob_survive_);
            auto predicted = filter_->predict(dt, bern->distribution());
            bern->set_distribution(predicted.clone_typed());
        }

        // Add birth Bernoullis to every global hypothesis
        std::vector<std::size_t> birth_indices;
        for (const auto& bb : birth_bernoullis_) {
            auto new_bern = bb->clone();
            new_bern->set_id(next_track_id_++);
            if (new_bern->has_distribution()) {
                increment_init_idx(new_bern->distribution());
            }
            std::size_t idx = bernoullis_.size();
            bernoullis_.push_back(std::move(new_bern));
            birth_indices.push_back(idx);
        }

        if (global_hypotheses_.empty()) {
            GlobalHypothesis h;
            h.log_weight = 0.0;
            h.bernoulli_indices = birth_indices;
            global_hypotheses_.push_back(std::move(h));
        } else {
            for (auto& hyp : global_hypotheses_) {
                for (auto idx : birth_indices) {
                    hyp.bernoulli_indices.push_back(idx);
                }
            }
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

        const double kappa_base = clutter_rate_ * clutter_density_;

        std::vector<double> kappa_vec(m);
        for (int j = 0; j < m; ++j) {
            const int W = static_cast<int>(meas_groups[j].cols());
            kappa_vec[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
        }

        std::vector<GlobalHypothesis> new_hypotheses;

        for (const auto& hyp : global_hypotheses_) {
            const int n_bern = static_cast<int>(hyp.bernoulli_indices.size());

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
                const auto& bern = *bernoullis_[hyp.bernoulli_indices[i]];
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

            auto solutions = assignment::murty(cost, k_best_);

            for (const auto& sol : solutions) {
                GlobalHypothesis new_hyp;
                new_hyp.log_weight = hyp.log_weight - sol.total_cost;

                std::vector<int> bern_to_meas(n_bern, -1);
                for (const auto& [row, col] : sol.assignments) {
                    if (row < n_bern && col < m) {
                        bern_to_meas[row] = col;
                    }
                }

                for (int i = 0; i < n_bern; ++i) {
                    std::size_t orig_idx = hyp.bernoulli_indices[i];
                    const auto& orig_bern = *bernoullis_[orig_idx];
                    double r = orig_bern.existence_probability();

                    if (bern_to_meas[i] >= 0) {
                        int j = bern_to_meas[i];
                        auto new_bern = std::make_unique<distributions::Bernoulli<T>>(
                            1.0, cache[i][j].dist->clone_typed(), orig_bern.id());
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(new_bern));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    } else {
                        double r_new = r * (1.0 - prob_detection_)
                                       / (1.0 - r * prob_detection_);
                        auto new_bern = std::make_unique<distributions::Bernoulli<T>>(
                            r_new, orig_bern.distribution().clone_typed(), orig_bern.id());
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(new_bern));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    }
                }

                new_hypotheses.push_back(std::move(new_hyp));
            }

            if (solutions.empty()) {
                GlobalHypothesis fallback;
                fallback.log_weight = hyp.log_weight;
                for (int i = 0; i < n_bern; ++i) {
                    std::size_t orig_idx = hyp.bernoulli_indices[i];
                    const auto& orig_bern = *bernoullis_[orig_idx];
                    double r = orig_bern.existence_probability();
                    double r_new = r * (1.0 - prob_detection_)
                                   / (1.0 - r * prob_detection_);
                    auto new_bern = std::make_unique<distributions::Bernoulli<T>>(
                        r_new, orig_bern.distribution().clone_typed(), orig_bern.id());
                    std::size_t new_idx = bernoullis_.size();
                    bernoullis_.push_back(std::move(new_bern));
                    fallback.bernoulli_indices.push_back(new_idx);
                }
                new_hypotheses.push_back(std::move(fallback));
            }
        }

        global_hypotheses_ = std::move(new_hypotheses);
    }

    void cleanup() override {
        normalize_log_weights();
        prune_hypotheses();
        cap_hypotheses();

        // Within each hypothesis, prune low-existence Bernoullis
        for (auto& hyp : global_hypotheses_) {
            std::vector<std::size_t> kept;
            for (auto idx : hyp.bernoulli_indices) {
                if (bernoullis_[idx]->existence_probability() >= prune_threshold_bern_) {
                    kept.push_back(idx);
                }
            }
            hyp.bernoulli_indices = std::move(kept);
        }

        // Compact Bernoulli table to reclaim memory from unreferenced entries
        compact_bernoulli_table();

        normalize_log_weights();

        // Compute cardinality distribution
        compute_cardinality();

        // Record track states
        record_track_states();

        // Extract
        extracted_mixtures_.push_back(extract());
    }

    [[nodiscard]] std::unique_ptr<distributions::Mixture<T>> extract() const {
        auto result = std::make_unique<distributions::Mixture<T>>();

        if (global_hypotheses_.empty()) return result;

        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(global_hypotheses_.size()); ++i) {
            if (global_hypotheses_[i].log_weight > global_hypotheses_[best_idx].log_weight) {
                best_idx = i;
            }
        }

        const auto& best = global_hypotheses_[best_idx];
        for (auto idx : best.bernoulli_indices) {
            const auto& bern = *bernoullis_[idx];
            if (bern.existence_probability() >= extract_threshold_ && bern.has_distribution()) {
                result->add_component(
                    bern.distribution().clone_typed(),
                    bern.existence_probability());
            }
        }

        return result;
    }

private:
    void correct_no_measurements() {
        for (auto& hyp : global_hypotheses_) {
            std::vector<std::size_t> new_indices;
            for (auto idx : hyp.bernoulli_indices) {
                const auto& orig = *bernoullis_[idx];
                double r = orig.existence_probability();
                double r_new = r * (1.0 - prob_detection_)
                               / (1.0 - r * prob_detection_);
                auto new_bern = std::make_unique<distributions::Bernoulli<T>>(
                    r_new, orig.distribution().clone_typed(), orig.id());
                std::size_t new_idx = bernoullis_.size();
                bernoullis_.push_back(std::move(new_bern));
                new_indices.push_back(new_idx);
            }
            hyp.bernoulli_indices = std::move(new_indices);
        }
    }

    /// Remove unreferenced Bernoulli entries and remap hypothesis indices.
    void compact_bernoulli_table() {
        std::vector<bool> referenced(bernoullis_.size(), false);
        for (const auto& hyp : global_hypotheses_) {
            for (auto idx : hyp.bernoulli_indices) {
                referenced[idx] = true;
            }
        }

        std::vector<std::size_t> new_index(bernoullis_.size(), 0);
        std::vector<std::unique_ptr<distributions::Bernoulli<T>>> compacted;
        for (std::size_t i = 0; i < bernoullis_.size(); ++i) {
            if (referenced[i]) {
                new_index[i] = compacted.size();
                compacted.push_back(std::move(bernoullis_[i]));
            }
        }

        for (auto& hyp : global_hypotheses_) {
            for (auto& idx : hyp.bernoulli_indices) {
                idx = new_index[idx];
            }
        }

        bernoullis_ = std::move(compacted);
    }

    void normalize_log_weights() {
        if (global_hypotheses_.empty()) return;
        double max_lw = global_hypotheses_[0].log_weight;
        for (const auto& h : global_hypotheses_) {
            max_lw = std::max(max_lw, h.log_weight);
        }
        double sum_exp = 0.0;
        for (auto& h : global_hypotheses_) {
            sum_exp += std::exp(h.log_weight - max_lw);
        }
        double log_norm = max_lw + std::log(sum_exp);
        for (auto& h : global_hypotheses_) {
            h.log_weight -= log_norm;
        }
    }

    void prune_hypotheses() {
        double log_thresh = std::log(prune_threshold_hyp_);
        std::vector<GlobalHypothesis> kept;
        for (auto& h : global_hypotheses_) {
            if (h.log_weight >= log_thresh) {
                kept.push_back(std::move(h));
            }
        }
        if (kept.empty() && !global_hypotheses_.empty()) {
            auto best = std::max_element(global_hypotheses_.begin(), global_hypotheses_.end(),
                [](const auto& a, const auto& b) { return a.log_weight < b.log_weight; });
            kept.push_back(std::move(*best));
        }
        global_hypotheses_ = std::move(kept);
    }

    void cap_hypotheses() {
        if (static_cast<int>(global_hypotheses_.size()) <= max_hypotheses_) return;
        std::sort(global_hypotheses_.begin(), global_hypotheses_.end(),
            [](const auto& a, const auto& b) { return a.log_weight > b.log_weight; });
        global_hypotheses_.resize(max_hypotheses_);
    }

    /// Compute cardinality PMF from component weights and label set sizes.
    void compute_cardinality() {
        if (global_hypotheses_.empty()) {
            cardinality_pmf_ = Eigen::VectorXd::Zero(1);
            estimated_cardinality_ = 0.0;
            return;
        }

        // Find max cardinality across all hypotheses
        int max_card = 0;
        for (const auto& hyp : global_hypotheses_) {
            // Count Bernoullis with r >= extract_threshold as "existing" targets
            int card = 0;
            for (auto idx : hyp.bernoulli_indices) {
                if (bernoullis_[idx]->existence_probability() >= extract_threshold_) {
                    card++;
                }
            }
            max_card = std::max(max_card, card);
        }

        cardinality_pmf_ = Eigen::VectorXd::Zero(max_card + 1);

        for (const auto& hyp : global_hypotheses_) {
            double w = std::exp(hyp.log_weight);
            int card = 0;
            for (auto idx : hyp.bernoulli_indices) {
                if (bernoullis_[idx]->existence_probability() >= extract_threshold_) {
                    card++;
                }
            }
            cardinality_pmf_(card) += w;
        }

        // Normalize
        double sum = cardinality_pmf_.sum();
        if (sum > 0.0) cardinality_pmf_ /= sum;

        // Compute mean
        estimated_cardinality_ = 0.0;
        for (int n = 0; n < cardinality_pmf_.size(); ++n) {
            estimated_cardinality_ += n * cardinality_pmf_(n);
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
        if (global_hypotheses_.empty()) return;

        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(global_hypotheses_.size()); ++i) {
            if (global_hypotheses_[i].log_weight > global_hypotheses_[best_idx].log_weight)
                best_idx = i;
        }

        const auto& best = global_hypotheses_[best_idx];
        for (auto idx : best.bernoulli_indices) {
            const auto& bern = *bernoullis_[idx];
            if (bern.existence_probability() >= extract_threshold_ && bern.has_distribution()) {
                track_histories_[bern.id()].push_back(get_track_state(bern.distribution()));
            }
        }
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    std::vector<std::unique_ptr<distributions::Bernoulli<T>>> bernoullis_;
    std::vector<GlobalHypothesis> global_hypotheses_;
    std::vector<std::unique_ptr<distributions::Bernoulli<T>>> birth_bernoullis_;

    double prune_threshold_hyp_ = 1e-3;
    double prune_threshold_bern_ = 1e-3;
    int max_hypotheses_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    int k_best_ = 5;
    int next_track_id_ = 0;
    bool is_extended_ = false;
    std::shared_ptr<clustering::DBSCAN> cluster_obj_;
    std::vector<std::unique_ptr<distributions::Mixture<T>>> extracted_mixtures_;
    std::map<int, std::vector<Eigen::VectorXd>> track_histories_;
    Eigen::VectorXd cardinality_pmf_;
    double estimated_cardinality_ = 0.0;
};

} // namespace brew::multi_target
