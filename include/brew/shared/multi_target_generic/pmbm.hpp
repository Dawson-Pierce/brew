#pragma once

#include "brew/shared/multi_target_generic/mbm_base.hpp"
#include "brew/assignment/murty.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/merge.hpp"
#include "brew/shared/fusion/cap.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace brew::multi_target {

/// Poisson Multi-Bernoulli Mixture (PMBM) filter. Maintains:
///   - A Poisson intensity for undetected targets (inherits from MBMBase's is_extended etc).
///   - A set of labeled Bernoulli tracks (from MBMBase::track_tab_) for detected targets.
/// Birth is implicit: the birth Mixture is added to the Poisson intensity at each predict.
/// The correct step runs an augmented Murty assignment where existing tracks, Poisson
/// spawn candidates (one per measurement), and clutter compete jointly.
// @mex rfs
// @mex_name PMBM
// @mex_params prune_poisson_threshold:double:1e-4, merge_poisson_threshold:double:4.0, max_poisson_components:int:100, prune_threshold_hypothesis:double:1e-3, prune_threshold_bernoulli:double:1e-3, recycle_threshold:double:0.1, max_hypotheses:int:100, extract_threshold:double:0.5, gate_threshold:double:9.0, k_best:int:5, gating_on:bool:false
// @mex_init set_poisson_intensity
// @mex_has cardinality, track_histories, cluster_object
template <typename T, int MaxComponents = Eigen::Dynamic>
class PMBM : public MBMBase<T, MaxComponents> {
public:
    using Base = MBMBase<T, MaxComponents>;
    using TrackEntry = typename Base::TrackEntry;
    using Hypothesis = typename Base::Hypothesis;
    using MixturePtr = typename Base::MixturePtr;

    PMBM() : Base() {
        Hypothesis h0;
        h0.log_weight = 0.0;
        this->hypotheses_.push_back(std::move(h0));
    }

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<PMBM<T, MaxComponents>>();
        c->hypotheses_.clear();
        c->prob_detection_ = this->prob_detection_;
        c->prob_survive_ = this->prob_survive_;
        c->clutter_rate_ = this->clutter_rate_;
        c->clutter_density_ = this->clutter_density_;
        if (this->filter_) c->filter_ = this->filter_->clone();
        c->has_filter_ = this->has_filter_;
        if (poisson_intensity_) c->poisson_intensity_ = poisson_intensity_->clone();
        if (birth_model_) c->birth_model_ = birth_model_->clone();
        for (const auto& t : this->track_tab_) c->track_tab_.push_back(t->clone());
        c->hypotheses_ = this->hypotheses_;
        c->prune_poisson_threshold_ = prune_poisson_threshold_;
        c->merge_poisson_threshold_ = merge_poisson_threshold_;
        c->max_poisson_components_ = max_poisson_components_;
        c->prune_threshold_hyp_ = this->prune_threshold_hyp_;
        c->prune_threshold_bern_ = this->prune_threshold_bern_;
        c->recycle_threshold_ = recycle_threshold_;
        c->max_hypotheses_ = this->max_hypotheses_;
        c->extract_threshold_ = this->extract_threshold_;
        c->gate_threshold_ = this->gate_threshold_;
        c->gating_on_ = this->gating_on_;
        c->k_best_ = this->k_best_;
        c->next_track_id_ = this->next_track_id_;
        c->is_extended_ = this->is_extended_;
        if (this->cluster_obj_) c->cluster_obj_ = this->cluster_obj_;
        for (const auto& e : this->extracted_hists_) {
            auto ne = std::make_unique<typename Base::ExtractedTrack>();
            ne->id = e->id;
            ne->b_time_index = e->b_time_index;
            ne->meas_assoc_hist = e->meas_assoc_hist;
            ne->states.reserve(e->states.size());
            for (const auto& s : e->states) ne->states.push_back(s->clone_typed());
            c->extracted_hists_.push_back(std::move(ne));
        }
        for (const auto& m : this->extracted_mixtures_) c->extracted_mixtures_.push_back(m->clone());
        c->cardinality_pmf_ = this->cardinality_pmf_;
        c->estimated_cardinality_ = this->estimated_cardinality_;
        c->time_index_cntr_ = this->time_index_cntr_;
        return c;
    }

    // ---- Configuration ----

    /// Poisson birth intensity added to the Poisson intensity each predict step.
    void set_birth_model(MixturePtr birth) {
        if (birth && !birth->empty()) {
            this->is_extended_ = birth->component(0).is_extended();
            if (this->is_extended_ && !this->cluster_obj_) {
                this->cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
        birth_model_ = std::move(birth);
    }

    /// Initial Poisson intensity (undetected targets).
    void set_poisson_intensity(MixturePtr intensity) {
        poisson_intensity_ = std::move(intensity);
    }

    void set_prune_poisson_threshold(double t) { prune_poisson_threshold_ = t; }
    void set_merge_poisson_threshold(double t) { merge_poisson_threshold_ = t; }
    void set_max_poisson_components(int n) { max_poisson_components_ = n; }
    void set_recycle_threshold(double t) { recycle_threshold_ = t; }

    // ---- Accessors ----

    [[nodiscard]] const models::Mixture<T, MaxComponents>& poisson_intensity() const {
        return *poisson_intensity_;
    }

    // ---- RFS interface ----

    void predict(int /*timestep*/, double dt) override {
        if (!this->has_filter_) return;

        // Poisson prediction (weight *= ps, each component predicted).
        if (poisson_intensity_) {
            for (std::size_t k = 0; k < poisson_intensity_->size(); ++k) {
                poisson_intensity_->weights()(static_cast<Eigen::Index>(k)) *= this->prob_survive_;
            }
            this->filter_->predict_batch_dynamic(dt, *poisson_intensity_);
            // Birth adds to Poisson intensity.
            if (birth_model_) {
                poisson_intensity_->add_components(*birth_model_);
            }
        }

        // Existing tracks: predict the latest mixture and append to history.
        for (auto& trk : this->track_tab_) {
            trk->existence_probability *= this->prob_survive_;
            trk->mixture_hist.push_back(
                this->predict_mixture(trk->current_mixture(), dt));
        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!this->has_filter_ || !poisson_intensity_) return;

        auto meas_groups = this->group_measurements(measurements);
        const int m = static_cast<int>(meas_groups.size());
        const double kappa_base = this->clutter_rate_ * this->clutter_density_;
        auto kappa_vec = Base::compute_kappa_vec(meas_groups, kappa_base);

        // Per-measurement Poisson spawn: corrected mixture and total likelihood weight.
        struct SpawnCandidate {
            MixturePtr mixture;       // normalized spatial mixture after update
            double total_weight = 0.0; // pd * sum_k w_k * qz_kj
        };
        std::vector<SpawnCandidate> spawn(m);

        for (int j = 0; j < m; ++j) {
            auto meas_flat = Base::flatten_measurement(meas_groups[j]);
            Eigen::VectorXd z_gate = Base::gate_center(meas_groups[j], this->is_extended_);

            auto mix = std::make_unique<models::Mixture<T, MaxComponents>>();
            std::vector<double> new_w;
            double total_w = 0.0;
            for (std::size_t k = 0; k < poisson_intensity_->size(); ++k) {
                if (this->gating_on_ &&
                    this->filter_->gate(z_gate, poisson_intensity_->component(k))
                        >= this->gate_threshold_) continue;
                auto [dist, qz] = this->filter_->correct(meas_flat,
                    poisson_intensity_->component(k));
                double w = this->prob_detection_ * poisson_intensity_->weight(k) * qz;
                if (w <= 0.0) continue;
                mix->add_component(dist.clone_typed(), w);
                new_w.push_back(w);
                total_w += w;
            }
            if (total_w > 0.0) {
                for (std::size_t c = 0; c < mix->size(); ++c) {
                    mix->weights()(static_cast<Eigen::Index>(c)) = new_w[c] / total_w;
                }
                spawn[j].mixture = std::move(mix);
                spawn[j].total_weight = total_w;
            }
        }

        // Remaining Poisson becomes undetected contribution.
        auto undetected = poisson_intensity_->clone();
        for (std::size_t k = 0; k < undetected->size(); ++k) {
            undetected->weights()(static_cast<Eigen::Index>(k)) *= (1.0 - this->prob_detection_);
        }

        // ---- Augmented Murty per parent hypothesis ----
        std::vector<Hypothesis> new_hyps;

        for (const auto& hyp : this->hypotheses_) {
            const int nt = static_cast<int>(hyp.num_tracks());

            if (m == 0) {
                // No measurements: all tracks get missed-detection update.
                Hypothesis h;
                h.log_weight = hyp.log_weight;
                for (int i = 0; i < nt; ++i) {
                    std::size_t orig_idx = hyp.track_indices[i];
                    const auto& orig = *this->track_tab_[orig_idx];
                    double r = orig.existence_probability;
                    double r_new = r * (1.0 - this->prob_detection_)
                                 / std::max(1.0 - r * this->prob_detection_,
                                            std::numeric_limits<double>::min());
                    auto child = orig.clone();
                    child->existence_probability = r_new;
                    child->meas_assoc_hist.push_back(-1);
                    std::size_t new_idx = this->track_tab_.size();
                    this->track_tab_.push_back(std::move(child));
                    h.track_indices.push_back(new_idx);
                }
                new_hyps.push_back(std::move(h));
                continue;
            }

            // Cost matrix layout:
            //   rows [0, nt):     existing tracks
            //   rows [nt, nt+m):  Poisson-spawn rows, one per measurement
            //   cols [0, m):      measurement j
            //   cols [m, m+nt+m): per-row miss diagonal
            const int n_rows = nt + m;
            const int n_cols = m + n_rows;
            Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(n_rows, n_cols,
                std::numeric_limits<double>::infinity());

            struct Cache { MixturePtr mix; double likelihood = 0.0; };
            std::vector<std::vector<Cache>> cache(nt);
            for (int i = 0; i < nt; ++i) cache[i].resize(m);

            // Existing track rows
            for (int i = 0; i < nt; ++i) {
                const auto& trk = *this->track_tab_[hyp.track_indices[i]];
                double r = trk.existence_probability;
                if (trk.mixture_hist.empty() || r <= 0.0) continue;

                for (int j = 0; j < m; ++j) {
                    Eigen::VectorXd z_gate = Base::gate_center(meas_groups[j], this->is_extended_);
                    if (this->gating_on_ && !this->passes_gate(trk, z_gate)) continue;
                    auto meas_flat = Base::flatten_measurement(meas_groups[j]);
                    auto [corrected, qz] = this->correct_mixture(trk.current_mixture(), meas_flat);
                    cache[i][j].mix = std::move(corrected);
                    cache[i][j].likelihood = qz;

                    double det = this->prob_detection_ * r * qz;
                    if (det > 0.0 && kappa_vec[j] > 0.0) {
                        cost(i, j) = -std::log(det / kappa_vec[j]);
                    }
                }

                double miss = 1.0 - this->prob_detection_ * r;
                if (miss > 0.0) cost(i, m + i) = -std::log(miss);
            }

            // Poisson-spawn rows
            for (int p = 0; p < m; ++p) {
                int row = nt + p;
                if (spawn[p].total_weight > 0.0 && kappa_vec[p] > 0.0) {
                    cost(row, p) = -std::log(spawn[p].total_weight / kappa_vec[p]);
                }
                // Spawn "miss" (no new track spawned) is free -- clutter baseline covers it.
                cost(row, m + row) = 0.0;
            }

            auto solutions = assignment::murty(cost, this->k_best_);

            for (const auto& sol : solutions) {
                Hypothesis new_hyp;
                new_hyp.log_weight = hyp.log_weight - sol.total_cost;

                std::vector<int> assign(n_rows, -1);
                for (const auto& [row, col] : sol.assignments) {
                    if (row < n_rows && col < m) assign[row] = col;
                }

                // Existing tracks: detected or missed.
                for (int i = 0; i < nt; ++i) {
                    std::size_t orig_idx = hyp.track_indices[i];
                    const auto& orig = *this->track_tab_[orig_idx];
                    auto child = orig.clone();
                    if (assign[i] >= 0) {
                        int j = assign[i];
                        child->mixture_hist.back() = cache[i][j].mix->clone();
                        child->existence_probability = 1.0;
                        child->meas_assoc_hist.push_back(j);
                    } else {
                        double r = orig.existence_probability;
                        child->existence_probability =
                            r * (1.0 - this->prob_detection_)
                            / std::max(1.0 - r * this->prob_detection_,
                                       std::numeric_limits<double>::min());
                        child->meas_assoc_hist.push_back(-1);
                    }
                    std::size_t new_idx = this->track_tab_.size();
                    this->track_tab_.push_back(std::move(child));
                    new_hyp.track_indices.push_back(new_idx);
                }

                // Poisson-spawn rows that fired: new Bernoulli tracks.
                for (int p = 0; p < m; ++p) {
                    int row = nt + p;
                    if (assign[row] < 0) continue;  // spawn did not fire
                    if (!spawn[p].mixture) continue;

                    // Existence conditioned on "target vs clutter" for this measurement.
                    double r_new = spawn[p].total_weight
                                 / (kappa_vec[p] + spawn[p].total_weight);

                    auto child = std::make_unique<TrackEntry>();
                    child->id = this->next_track_id_++;
                    child->birth_time_index = this->time_index_cntr_;
                    child->existence_probability = r_new;
                    child->mixture_hist.push_back(spawn[p].mixture->clone());
                    child->meas_assoc_hist.push_back(p);
                    std::size_t new_idx = this->track_tab_.size();
                    this->track_tab_.push_back(std::move(child));
                    new_hyp.track_indices.push_back(new_idx);
                }

                new_hyps.push_back(std::move(new_hyp));
            }
        }

        this->hypotheses_ = std::move(new_hyps);
        poisson_intensity_ = std::move(undetected);
    }

    void cleanup() override {
        // Poisson cleanup.
        if (poisson_intensity_) {
            fusion::prune(*poisson_intensity_, prune_poisson_threshold_);
            fusion::merge(*poisson_intensity_, merge_poisson_threshold_);
            fusion::cap(*poisson_intensity_,
                        static_cast<std::size_t>(max_poisson_components_));
        }

        this->normalize_log_weights();
        this->prune_hypotheses();
        this->cap_hypotheses();
        recycle_low_existence_to_poisson();
        this->compact_track_table();
        this->normalize_log_weights();
        this->compute_cardinality();
        this->extract_states();
        this->push_extracted_snapshot();
        ++this->time_index_cntr_;
    }

private:
    /// Recycle tracks whose existence is between prune_threshold and recycle_threshold
    /// back into the Poisson intensity. Above recycle_threshold: keep as Bernoulli.
    /// Below prune_threshold: drop entirely.
    void recycle_low_existence_to_poisson() {
        std::vector<bool> recycled(this->track_tab_.size(), false);
        for (auto& hyp : this->hypotheses_) {
            std::vector<std::size_t> kept;
            kept.reserve(hyp.track_indices.size());
            for (auto idx : hyp.track_indices) {
                double r = this->track_tab_[idx]->existence_probability;
                if (r < this->prune_threshold_bern_) continue;
                if (r < recycle_threshold_ && poisson_intensity_
                    && !recycled[idx]
                    && !this->track_tab_[idx]->mixture_hist.empty())
                {
                    recycled[idx] = true;
                    const auto& last_mix = this->track_tab_[idx]->current_mixture();
                    for (std::size_t c = 0; c < last_mix.size(); ++c) {
                        poisson_intensity_->add_component(
                            last_mix.component(c).clone_typed(),
                            r * last_mix.weight(c));
                    }
                    continue;
                }
                kept.push_back(idx);
            }
            hyp.track_indices = std::move(kept);
        }
    }

    // ---- Members ----

    std::unique_ptr<models::Mixture<T, MaxComponents>> poisson_intensity_;
    std::unique_ptr<models::Mixture<T, MaxComponents>> birth_model_;

    double prune_poisson_threshold_ = 1e-4;
    double merge_poisson_threshold_ = 4.0;
    int max_poisson_components_ = 100;
    double recycle_threshold_ = 0.1;
};

}  // namespace brew::multi_target
