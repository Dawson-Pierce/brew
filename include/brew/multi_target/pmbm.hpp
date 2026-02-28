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
#include <limits>
#include <type_traits>

namespace brew::multi_target {

/// Poisson Multi-Bernoulli Mixture (PMBM) filter.
/// Combines a Poisson intensity (for undetected targets) with an MBM component
/// (for detected targets). On first detection, Poisson components spawn new
/// Bernoulli tracks. This is the theoretically optimal conjugate prior for the
/// multi-target tracking problem.
/// Template parameter T is the single distribution type (e.g., Gaussian, GGIW).
template <typename T>
class PMBM : public RFSBase {
    /// Internal global hypothesis type
    struct GlobalHypothesis_ {
        double log_weight = 0.0;
        std::vector<std::size_t> bernoulli_indices;
    };

public:
    PMBM() = default;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<PMBM<T>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        if (poisson_intensity_) c->poisson_intensity_ = poisson_intensity_->clone();
        if (birth_model_) c->birth_model_ = birth_model_->clone();
        for (const auto& b : bernoullis_)
            c->bernoullis_.push_back(b->clone());
        c->global_hypotheses_ = global_hypotheses_;
        c->prune_poisson_threshold_ = prune_poisson_threshold_;
        c->merge_poisson_threshold_ = merge_poisson_threshold_;
        c->max_poisson_components_ = max_poisson_components_;
        c->prune_threshold_hyp_ = prune_threshold_hyp_;
        c->prune_threshold_bern_ = prune_threshold_bern_;
        c->recycle_threshold_ = recycle_threshold_;
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

    /// Set the Poisson birth model (added to Poisson intensity each predict step).
    void set_birth_model(std::unique_ptr<models::Mixture<T>> birth) {
        if (birth && !birth->empty()) {
            is_extended_ = birth->component(0).is_extended();
            if (is_extended_ && !cluster_obj_) {
                cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
        birth_model_ = std::move(birth);
    }

    /// Set the initial Poisson intensity (undetected targets).
    void set_poisson_intensity(std::unique_ptr<models::Mixture<T>> intensity) {
        poisson_intensity_ = std::move(intensity);
    }

    void set_prune_poisson_threshold(double t) { prune_poisson_threshold_ = t; }
    void set_merge_poisson_threshold(double t) { merge_poisson_threshold_ = t; }
    void set_max_poisson_components(int n) { max_poisson_components_ = n; }
    void set_prune_threshold_hypothesis(double t) { prune_threshold_hyp_ = t; }
    void set_prune_threshold_bernoulli(double t) { prune_threshold_bern_ = t; }
    void set_recycle_threshold(double t) { recycle_threshold_ = t; }
    void set_max_hypotheses(int n) { max_hypotheses_ = n; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_k_best(int k) { k_best_ = k; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::DBSCAN> obj) {
        cluster_obj_ = std::move(obj);
    }

    // ---- Accessors ----

    [[nodiscard]] const models::Mixture<T>& poisson_intensity() const {
        return *poisson_intensity_;
    }

    [[nodiscard]] const std::vector<GlobalHypothesis_>& global_hypotheses() const {
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

        // --- Poisson intensity prediction (same as PHD) ---
        if (poisson_intensity_) {
            for (std::size_t k = 0; k < poisson_intensity_->size(); ++k) {
                poisson_intensity_->weights()(static_cast<Eigen::Index>(k)) *= prob_survive_;
                auto predicted = filter_->predict(dt, poisson_intensity_->component(k));
                *poisson_intensity_->components()[k] = std::move(predicted);
            }

            // Add birth to Poisson intensity
            if (birth_model_) {
                auto birth_copy = birth_model_->clone();
                poisson_intensity_->add_components(*birth_copy);

                for (std::size_t k = 0; k < birth_model_->size(); ++k) {
                    increment_init_idx(birth_model_->component(k));
                }
            }
        }

        // --- MBM Bernoulli prediction ---
        for (auto& bern : bernoullis_) {
            if (!bern->has_distribution()) continue;
            bern->set_existence_probability(bern->existence_probability() * prob_survive_);
            auto predicted = filter_->predict(dt, bern->distribution());
            bern->set_distribution(predicted.clone_typed());
        }

        // Initialize a single empty hypothesis if none exist
        if (global_hypotheses_.empty()) {
            GlobalHypothesis_ h;
            h.log_weight = 0.0;
            global_hypotheses_.push_back(std::move(h));
        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!filter_ || !poisson_intensity_) return;

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
        const double kappa_base = clutter_rate_ * clutter_density_;

        // Pre-compute per-cluster clutter term: for a cluster of W measurements,
        // the clutter likelihood is the product of independent clutter intensities.
        std::vector<double> kappa_vec(m);
        for (int j = 0; j < m; ++j) {
            const int W = static_cast<int>(meas_groups[j].cols());
            kappa_vec[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
        }

        // --- Create new Bernoulli tracks from Poisson for each measurement ---
        // For each measurement, compute the likelihood against all Poisson components
        // and create a new Bernoulli representing "first detection from undetected pool"
        struct NewTrack {
            std::unique_ptr<models::Bernoulli<T>> bernoulli;
            double log_likelihood = 0.0; // log of total detection weight from Poisson
        };
        std::vector<NewTrack> new_tracks(m);

        for (int j = 0; j < m; ++j) {
            const Eigen::MatrixXd& meas = meas_groups[j];
            Eigen::VectorXd z_gate = (is_extended_ && meas.cols() > 1)
                ? Eigen::VectorXd(meas.rowwise().mean())
                : Eigen::VectorXd(meas.col(0));

            Eigen::VectorXd meas_flat;
            if (meas.cols() > 1) {
                meas_flat.resize(meas.size());
                for (int c = 0; c < meas.cols(); ++c)
                    meas_flat.segment(c * meas.rows(), meas.rows()) = meas.col(c);
            } else {
                meas_flat = meas.col(0);
            }

            // Accumulate corrected distributions weighted by Poisson weights
            double total_weight = 0.0;
            std::unique_ptr<T> best_dist;
            double best_weight = -1.0;

            for (std::size_t k = 0; k < poisson_intensity_->size(); ++k) {
                double gate_val = filter_->gate(z_gate, poisson_intensity_->component(k));
                if (gate_val < gate_threshold_) {
                    auto [dist, qz] = filter_->correct(meas_flat, poisson_intensity_->component(k));
                    double w = prob_detection_ * poisson_intensity_->weight(k) * qz;
                    total_weight += w;
                    if (w > best_weight) {
                        best_weight = w;
                        best_dist = dist.clone_typed();
                    }
                }
            }

            const double kappa_j = kappa_vec[j];
            if (total_weight > 0.0 && best_dist) {
                // Existence probability for new Bernoulli from Poisson
                double r_new = total_weight / (kappa_j + total_weight);
                auto bern = std::make_unique<models::Bernoulli<T>>(
                    r_new, std::move(best_dist), next_track_id_++);
                new_tracks[j].bernoulli = std::move(bern);
                new_tracks[j].log_likelihood = std::log(kappa_j + total_weight);
            } else {
                new_tracks[j].log_likelihood = std::log(std::max(kappa_j, 1e-300));
            }
        }

        // --- Scale Poisson intensity by (1 - p_D) for undetected ---
        for (std::size_t k = 0; k < poisson_intensity_->size(); ++k) {
            poisson_intensity_->weights()(static_cast<Eigen::Index>(k)) *= (1.0 - prob_detection_);
        }

        // --- MBM update: for each hypothesis, build cost matrix including existing
        //     Bernoullis and new tracks from Poisson ---
        std::vector<GlobalHypothesis_> new_hypotheses;

        for (const auto& hyp : global_hypotheses_) {
            const int n_bern = static_cast<int>(hyp.bernoulli_indices.size());

            if (m == 0) {
                // No measurements: missed detection for all Bernoullis
                GlobalHypothesis_ new_hyp;
                new_hyp.log_weight = hyp.log_weight;
                for (int i = 0; i < n_bern; ++i) {
                    std::size_t orig_idx = hyp.bernoulli_indices[i];
                    const auto& orig = *bernoullis_[orig_idx];
                    double r = orig.existence_probability();
                    double r_new = r * (1.0 - prob_detection_) / (1.0 - r * prob_detection_);
                    auto nb = std::make_unique<models::Bernoulli<T>>(
                        r_new, orig.distribution().clone_typed(), orig.id());
                    std::size_t new_idx = bernoullis_.size();
                    bernoullis_.push_back(std::move(nb));
                    new_hyp.bernoulli_indices.push_back(new_idx);
                }
                new_hypotheses.push_back(std::move(new_hyp));
                continue;
            }

            // Cost matrix: n_bern rows x (m + n_bern) cols
            // Rows = existing Bernoullis
            // Cols 0..m-1 = measurement associations
            // Cols m..m+n_bern-1 = missed detection (diagonal)
            const int n_cols = m + n_bern;
            Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(
                n_bern, n_cols, std::numeric_limits<double>::infinity());

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

                    if (filter_->gate(z_gate, bern.distribution()) < gate_threshold_) {
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

                // Missed detection
                double miss_factor = 1.0 - prob_detection_ * r;
                if (miss_factor > 0.0) {
                    cost(i, m + i) = -std::log(miss_factor);
                }
                // else: leave at infinity — missing a certainly-detected target is forbidden
            }

            // Solve K-best assignments
            auto solutions = (n_bern > 0) ? assignment::murty(cost, k_best_)
                                           : std::vector<assignment::AssignmentResult>{};

            // If no Bernoullis, create a single "all missed" solution
            if (n_bern == 0) {
                assignment::AssignmentResult empty_sol;
                empty_sol.total_cost = 0.0;
                solutions.push_back(empty_sol);
            }

            for (const auto& sol : solutions) {
                GlobalHypothesis_ new_hyp;
                new_hyp.log_weight = hyp.log_weight - sol.total_cost;

                // Determine Bernoulli-to-measurement mapping
                std::vector<int> bern_to_meas(n_bern, -1);
                std::vector<bool> meas_assigned(m, false);
                for (const auto& [row, col] : sol.assignments) {
                    if (row < n_bern && col < m) {
                        bern_to_meas[row] = col;
                        meas_assigned[col] = true;
                    }
                }

                // Update existing Bernoullis
                for (int i = 0; i < n_bern; ++i) {
                    std::size_t orig_idx = hyp.bernoulli_indices[i];
                    const auto& orig = *bernoullis_[orig_idx];
                    double r = orig.existence_probability();

                    if (bern_to_meas[i] >= 0) {
                        int j = bern_to_meas[i];
                        auto nb = std::make_unique<models::Bernoulli<T>>(
                            1.0, cache[i][j].dist->clone_typed(), orig.id());
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(nb));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    } else {
                        double r_new = r * (1.0 - prob_detection_) / (1.0 - r * prob_detection_);
                        auto nb = std::make_unique<models::Bernoulli<T>>(
                            r_new, orig.distribution().clone_typed(), orig.id());
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(nb));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    }
                }

                // Unassigned measurements: spawn new Bernoulli from Poisson
                for (int j = 0; j < m; ++j) {
                    if (!meas_assigned[j] && new_tracks[j].bernoulli) {
                        auto nb = new_tracks[j].bernoulli->clone();
                        std::size_t new_idx = bernoullis_.size();
                        bernoullis_.push_back(std::move(nb));
                        new_hyp.bernoulli_indices.push_back(new_idx);
                    }
                    // Add log-likelihood ratio vs clutter baseline
                    if (!meas_assigned[j]) {
                        new_hyp.log_weight += new_tracks[j].log_likelihood
                                              - std::log(kappa_vec[j]);
                    }
                }

                new_hypotheses.push_back(std::move(new_hyp));
            }
        }

        global_hypotheses_ = std::move(new_hypotheses);
    }

    void cleanup() override {
        // --- Poisson cleanup ---
        if (poisson_intensity_) {
            fusion::prune(*poisson_intensity_, prune_poisson_threshold_);
            fusion::merge(*poisson_intensity_, merge_poisson_threshold_);
            fusion::cap(*poisson_intensity_, static_cast<std::size_t>(max_poisson_components_));
        }

        // --- Hypothesis cleanup ---
        normalize_log_weights();
        prune_hypotheses();
        cap_hypotheses();

        // Prune low-existence Bernoullis and recycle to Poisson.
        // Track recycled indices to avoid double-counting across hypotheses.
        std::vector<bool> recycled(bernoullis_.size(), false);
        for (auto& hyp : global_hypotheses_) {
            std::vector<std::size_t> kept;
            for (auto idx : hyp.bernoulli_indices) {
                double r = bernoullis_[idx]->existence_probability();
                if (r >= prune_threshold_bern_) {
                    if (r < recycle_threshold_ && poisson_intensity_ &&
                        bernoullis_[idx]->has_distribution()) {
                        // Recycle: add back to Poisson intensity (only once per Bernoulli)
                        if (!recycled[idx]) {
                            recycled[idx] = true;
                            poisson_intensity_->add_component(
                                bernoullis_[idx]->distribution().clone_typed(), r);
                        }
                    } else {
                        kept.push_back(idx);
                    }
                }
            }
            hyp.bernoulli_indices = std::move(kept);
        }

        // Compact Bernoulli table to reclaim memory from unreferenced entries
        compact_bernoulli_table();

        normalize_log_weights();

        // Compute cardinality distribution
        compute_cardinality();

        // Record track state ancestry before extraction
        record_track_states();

        // Extract and store
        extracted_mixtures_.push_back(extract());
    }

    /// Extract state estimates from the best global hypothesis.
    [[nodiscard]] std::unique_ptr<models::Mixture<T>> extract() const {
        auto result = std::make_unique<models::Mixture<T>>();

        if (global_hypotheses_.empty()) return result;

        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(global_hypotheses_.size()); ++i) {
            if (global_hypotheses_[i].log_weight > global_hypotheses_[best_idx].log_weight)
                best_idx = i;
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
        for (const auto& h : global_hypotheses_)
            max_lw = std::max(max_lw, h.log_weight);
        double sum_exp = 0.0;
        for (auto& h : global_hypotheses_)
            sum_exp += std::exp(h.log_weight - max_lw);
        double log_norm = max_lw + std::log(sum_exp);
        for (auto& h : global_hypotheses_)
            h.log_weight -= log_norm;
    }

    void prune_hypotheses() {
        double log_thresh = std::log(prune_threshold_hyp_);
        std::vector<GlobalHypothesis_> kept;
        for (auto& h : global_hypotheses_) {
            if (h.log_weight >= log_thresh)
                kept.push_back(std::move(h));
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

    // Poisson component (undetected targets)
    std::unique_ptr<models::Mixture<T>> poisson_intensity_;
    std::unique_ptr<models::Mixture<T>> birth_model_;

    // MBM component (detected targets)
    std::vector<std::unique_ptr<models::Bernoulli<T>>> bernoullis_;
    std::vector<GlobalHypothesis_> global_hypotheses_;

    // Poisson parameters
    double prune_poisson_threshold_ = 1e-4;
    double merge_poisson_threshold_ = 4.0;
    int max_poisson_components_ = 100;

    // MBM parameters
    double prune_threshold_hyp_ = 1e-3;
    double prune_threshold_bern_ = 1e-3;
    double recycle_threshold_ = 0.1;
    int max_hypotheses_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    int k_best_ = 5;
    int next_track_id_ = 0;

    bool is_extended_ = false;
    std::shared_ptr<clustering::DBSCAN> cluster_obj_;
    std::vector<std::unique_ptr<models::Mixture<T>>> extracted_mixtures_;
    std::map<int, std::vector<Eigen::VectorXd>> track_histories_;
    Eigen::VectorXd cardinality_pmf_;
    double estimated_cardinality_ = 0.0;
};

} // namespace brew::multi_target
