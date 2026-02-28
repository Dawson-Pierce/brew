#pragma once

#include "brew/multi_target/rfs_base.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/bernoulli.hpp"
#include "brew/models/trajectory_base_model.hpp"
#include "brew/filters/filter.hpp"
#include "brew/fusion/prune.hpp"
#include "brew/fusion/merge.hpp"
#include "brew/fusion/cap.hpp"
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

/// A single global hypothesis: a weighted collection of Bernoulli indices.
struct GlobalHypothesis {
    double log_weight = 0.0;
    /// Indices into the MBM's bernoulli table for this hypothesis.
    std::vector<std::size_t> bernoulli_indices;
};

/// Multi-Bernoulli Mixture (MBM) filter.
/// Maintains explicit data association hypotheses via a mixture of
/// multi-Bernoulli components. Uses Murty's K-best algorithm for
/// hypothesis generation.
/// Template parameter T is the single distribution type (e.g., Gaussian, GGIW).
template <typename T>
class MBM : public RFSBase {
public:
    MBM() = default;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<MBM<T>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        // Clone all Bernoulli components
        for (const auto& b : bernoullis_) {
            c->bernoullis_.push_back(b->clone());
        }
        c->global_hypotheses_ = global_hypotheses_;
        // Clone birth Bernoullis
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
        return c;
    }

    // ---- Configuration ----

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
    }

    /// Set birth Bernoullis added at each predict step.
    void set_birth_bernoullis(std::vector<std::unique_ptr<models::Bernoulli<T>>> birth) {
        birth_bernoullis_ = std::move(birth);
    }

    /// Convenience: create birth Bernoullis from a Mixture (each component becomes
    /// a Bernoulli with existence_prob = weight).
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

    [[nodiscard]] const std::vector<std::unique_ptr<models::Mixture<T>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    [[nodiscard]] int num_tracks() const {
        return static_cast<int>(bernoullis_.size());
    }

    /// Per-track state history: maps track ID -> vector of state means over time.
    [[nodiscard]] const std::map<int, std::vector<Eigen::VectorXd>>& track_histories() const {
        return track_histories_;
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
            // Handle trajectory init_idx
            if (new_bern->has_distribution()) {
                increment_init_idx(new_bern->distribution());
            }
            std::size_t idx = bernoullis_.size();
            bernoullis_.push_back(std::move(new_bern));
            birth_indices.push_back(idx);
        }

        // Add birth indices to all existing hypotheses
        if (global_hypotheses_.empty()) {
            // First step: create a single hypothesis with just birth
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
            // No measurements: just apply missed detection to all Bernoullis
            correct_no_measurements();
            return;
        }

        const double kappa_base = clutter_rate_ * clutter_density_;

        // Pre-compute per-cluster clutter term: for a cluster of W measurements,
        // the clutter likelihood is the product of independent clutter intensities.
        std::vector<double> kappa_vec(m);
        for (int j = 0; j < m; ++j) {
            const int W = static_cast<int>(meas_groups[j].cols());
            kappa_vec[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
        }

        std::vector<GlobalHypothesis> new_hypotheses;

        for (const auto& hyp : global_hypotheses_) {
            const int n_bern = static_cast<int>(hyp.bernoulli_indices.size());

            // Build cost matrix: n_bern rows x (m + n_bern) cols
            // Cols 0..m-1: measurement associations
            // Cols m..m+n_bern-1: missed detection (diagonal)
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
                const auto& bern = *bernoullis_[hyp.bernoulli_indices[i]];
                double r = bern.existence_probability();
                if (!bern.has_distribution() || r <= 0.0) continue;

                // Measurement association costs
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

                        // Cost = -log(detection_likelihood / clutter_likelihood)
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
                }
                // else: leave at infinity — missing a certainly-detected target is forbidden
            }

            // Solve K-best assignments
            auto solutions = assignment::murty(cost, k_best_);

            for (const auto& sol : solutions) {
                GlobalHypothesis new_hyp;
                new_hyp.log_weight = hyp.log_weight - sol.total_cost;

                // Build updated Bernoulli set
                // First, determine which Bernoullis are detected vs missed
                std::vector<int> bern_to_meas(n_bern, -1); // -1 = missed
                for (const auto& [row, col] : sol.assignments) {
                    if (row < n_bern && col < m) {
                        bern_to_meas[row] = col;
                    }
                    // col >= m means missed detection (which is the default)
                }

                for (int i = 0; i < n_bern; ++i) {
                    std::size_t orig_idx = hyp.bernoulli_indices[i];
                    const auto& orig_bern = *bernoullis_[orig_idx];
                    double r = orig_bern.existence_probability();

                    if (bern_to_meas[i] >= 0) {
                        // Detected: create updated Bernoulli
                        int j = bern_to_meas[i];
                        auto new_bern = std::make_unique<models::Bernoulli<T>>(
                            1.0, // existence confirmed by detection
                            cache[i][j].dist->clone_typed(),
                            orig_bern.id());
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(new_bern));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    } else {
                        // Missed: update existence probability
                        double r_new = r * (1.0 - prob_detection_)
                                       / (1.0 - r * prob_detection_);
                        auto new_bern = std::make_unique<models::Bernoulli<T>>(
                            r_new,
                            orig_bern.distribution().clone_typed(),
                            orig_bern.id());
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(new_bern));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    }
                }

                new_hypotheses.push_back(std::move(new_hyp));
            }

            // If no solutions found (shouldn't happen), keep original with missed detection
            if (solutions.empty()) {
                GlobalHypothesis fallback;
                fallback.log_weight = hyp.log_weight;
                for (int i = 0; i < n_bern; ++i) {
                    std::size_t orig_idx = hyp.bernoulli_indices[i];
                    const auto& orig_bern = *bernoullis_[orig_idx];
                    double r = orig_bern.existence_probability();
                    double r_new = r * (1.0 - prob_detection_)
                                   / (1.0 - r * prob_detection_);
                    auto new_bern = std::make_unique<models::Bernoulli<T>>(
                        r_new,
                        orig_bern.distribution().clone_typed(),
                        orig_bern.id());
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
        // Normalize hypothesis log-weights
        normalize_log_weights();

        // Prune low-weight hypotheses
        prune_hypotheses();

        // Cap hypotheses
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

        // Re-normalize after pruning
        normalize_log_weights();

        // Record track state ancestry before extraction
        record_track_states();

        // Extract and store
        extracted_mixtures_.push_back(extract());
    }

    /// Extract state estimates from the best (MAP) global hypothesis.
    [[nodiscard]] std::unique_ptr<models::Mixture<T>> extract() const {
        auto result = std::make_unique<models::Mixture<T>>();

        if (global_hypotheses_.empty()) return result;

        // Find highest-weight hypothesis
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
        // All Bernoullis receive missed detection update
        for (auto& hyp : global_hypotheses_) {
            std::vector<std::size_t> new_indices;
            for (auto idx : hyp.bernoulli_indices) {
                const auto& orig = *bernoullis_[idx];
                double r = orig.existence_probability();
                double r_new = r * (1.0 - prob_detection_)
                               / (1.0 - r * prob_detection_);
                auto new_bern = std::make_unique<models::Bernoulli<T>>(
                    r_new,
                    orig.distribution().clone_typed(),
                    orig.id());
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
        std::vector<std::unique_ptr<models::Bernoulli<T>>> compacted;
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
            // Keep at least the best hypothesis
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

    // Trajectory init_idx helpers
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

    // SFINAE trait for trajectory-type distributions
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

    /// Record MMSE state of each track into history.
    void record_track_states() {
        if (global_hypotheses_.empty()) return;

        std::map<int, std::pair<Eigen::VectorXd, double>> accum;
        for (const auto& hyp : global_hypotheses_) {
            double w = std::exp(hyp.log_weight);
            for (auto idx : hyp.bernoulli_indices) {
                const auto& bern = *bernoullis_[idx];
                if (bern.existence_probability() >= extract_threshold_ && bern.has_distribution()) {
                    Eigen::VectorXd state = get_track_state(bern.distribution());
                    auto it = accum.find(bern.id());
                    if (it == accum.end()) {
                        accum[bern.id()] = {w * state, w};
                    } else {
                        it->second.first += w * state;
                        it->second.second += w;
                    }
                }
            }
        }

        for (const auto& [tid, ws] : accum) {
            track_histories_[tid].push_back(ws.first / ws.second);
        }
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    std::vector<std::unique_ptr<models::Bernoulli<T>>> bernoullis_; // shared table
    std::vector<GlobalHypothesis> global_hypotheses_;
    std::vector<std::unique_ptr<models::Bernoulli<T>>> birth_bernoullis_;

    double prune_threshold_hyp_ = 1e-3;
    double prune_threshold_bern_ = 1e-3;
    int max_hypotheses_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    int k_best_ = 5;
    int next_track_id_ = 0;
    bool is_extended_ = false;
    std::shared_ptr<clustering::DBSCAN> cluster_obj_;
    std::vector<std::unique_ptr<models::Mixture<T>>> extracted_mixtures_;
    std::map<int, std::vector<Eigen::VectorXd>> track_histories_;
};

} // namespace brew::multi_target
