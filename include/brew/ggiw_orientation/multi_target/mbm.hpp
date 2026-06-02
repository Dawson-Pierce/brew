#pragma once

#include "brew/ggiw_orientation/ggiw_orientation_model.hpp"

#include "brew/ggiw_orientation/multi_target/mbm_base.hpp"
#include "brew/assignment/murty.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace brew::ggiw_orientation {

/// Multi-Bernoulli Mixture (MBM) filter. Unlabeled. Each track is a Bernoulli with an
/// explicit existence probability and an internal mixture over single-model objects.
/// Birth is an LMB RFS with per-term spatial mixtures and explicit birth probabilities.
/// The MBM maintains weighted hypotheses over track subsets; each correct step uses
/// Murty's m-best assignment to enumerate likely hypotheses.
template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class MBM : public MBMBase<Scalar, D, De, MaxComponents> {
    using T = models::GGIWOrientation<Scalar, D, De>;
public:
    using Base = MBMBase<Scalar, D, De, MaxComponents>;
    using TrackEntry = typename Base::TrackEntry;
    using Hypothesis = typename Base::Hypothesis;
    using MixturePtr = typename Base::MixturePtr;

    MBM() : Base() {
        Hypothesis h0;
        h0.log_weight = 0.0;
        this->hypotheses_.push_back(std::move(h0));
    }

    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto c = std::make_unique<MBM<Scalar, D, De, MaxComponents>>();
        c->hypotheses_.clear();  // clear default init
        c->prob_detection_ = this->prob_detection_;
        c->prob_survive_ = this->prob_survive_;
        c->clutter_rate_ = this->clutter_rate_;
        c->clutter_density_ = this->clutter_density_;
        if (this->filter_) c->filter_ = this->filter_->clone();
        c->has_filter_ = this->has_filter_;
        for (const auto& t : this->track_tab_) c->track_tab_.push_back(t->clone());
        c->hypotheses_ = this->hypotheses_;
        for (const auto& bt : birth_terms_) {
            c->birth_terms_.emplace_back(bt.first->clone(), bt.second);
        }
        c->prune_threshold_hyp_ = this->prune_threshold_hyp_;
        c->prune_threshold_bern_ = this->prune_threshold_bern_;
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

    /// Birth model: a list of (spatial mixture, birth probability) terms.
    void set_birth_terms(std::vector<std::pair<MixturePtr, double>> terms) {
        birth_terms_ = std::move(terms);
        if (!birth_terms_.empty() && !birth_terms_[0].first->empty()) {
            this->is_extended_ = birth_terms_[0].first->component(0).is_extended();
            if (this->is_extended_ && !this->cluster_obj_) {
                this->cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
    }

    /// Convenience overload: one spatial mixture with a single birth probability.
    void set_birth_model(MixturePtr birth_mix, double p_birth) {
        birth_terms_.clear();
        if (!birth_mix || birth_mix->empty()) return;
        birth_terms_.emplace_back(std::move(birth_mix), p_birth);
        this->is_extended_ = birth_terms_[0].first->component(0).is_extended();
        if (this->is_extended_ && !this->cluster_obj_) {
            this->cluster_obj_ = std::make_shared<clustering::DBSCAN>();
        }
    }

    // ---- RFS interface ----

    void predict(int /*timestep*/, double dt) override {
        if (!this->has_filter_) return;

        // Propagate existing tracks (append predicted mixture to history).
        for (auto& trk : this->track_tab_) {
            trk->existence_probability *= this->prob_survive_;
            trk->mixture_hist.push_back(this->predict_mixture(trk->current_mixture(), dt));
        }

        // Spawn birth tracks and add them to every hypothesis.
        std::vector<std::size_t> birth_indices;
        for (std::size_t i = 0; i < birth_terms_.size(); ++i) {
            auto entry = std::make_unique<TrackEntry>();
            entry->id = this->next_track_id_++;
            entry->existence_probability = birth_terms_[i].second;
            entry->birth_time_index = this->time_index_cntr_;
            entry->mixture_hist.push_back(birth_terms_[i].first->clone());
            birth_indices.push_back(this->track_tab_.size());
            this->track_tab_.push_back(std::move(entry));
        }
        if (this->hypotheses_.empty()) {
            Hypothesis h;
            h.log_weight = 0.0;
            h.track_indices = birth_indices;
            this->hypotheses_.push_back(std::move(h));
        } else {
            for (auto& hyp : this->hypotheses_) {
                for (auto idx : birth_indices) hyp.track_indices.push_back(idx);
            }
        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!this->has_filter_) return;

        auto meas_groups = this->group_measurements(measurements);
        const int m = static_cast<int>(meas_groups.size());

        if (m == 0) {
            correct_no_measurements();
            return;
        }

        const double kappa_base = this->clutter_rate_ * this->clutter_density_;
        auto kappa_vec = Base::compute_kappa_vec(meas_groups, kappa_base);

        std::vector<Hypothesis> new_hypotheses;
        for (const auto& hyp : this->hypotheses_) {
            const int n = static_cast<int>(hyp.num_tracks());
            const int ncols = m + n;

            Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(n, ncols,
                std::numeric_limits<double>::infinity());

            struct Cache { MixturePtr mix; double likelihood = 0.0; };
            std::vector<std::vector<Cache>> cache(n);
            for (int i = 0; i < n; ++i) cache[i].resize(m);

            for (int i = 0; i < n; ++i) {
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
                // else: leaving miss at infinity forbids missing a certainly-detected track.
            }

            auto solutions = (n > 0)
                ? assignment::murty(cost, this->k_best_)
                : std::vector<assignment::AssignmentResult>{};

            if (n == 0) {
                // No existing tracks: single "all clutter" continuation.
                assignment::AssignmentResult empty;
                empty.total_cost = 0.0;
                solutions.push_back(empty);
            }

            for (const auto& sol : solutions) {
                Hypothesis new_hyp;
                new_hyp.log_weight = hyp.log_weight - sol.total_cost;

                std::vector<int> assign(n, -1);
                for (const auto& [row, col] : sol.assignments) {
                    if (row < n && col < m) assign[row] = col;
                }

                for (int i = 0; i < n; ++i) {
                    std::size_t orig_idx = hyp.track_indices[i];
                    const auto& orig = *this->track_tab_[orig_idx];
                    auto child = orig.clone();
                    if (assign[i] >= 0) {
                        // Detected: corrected mixture, existence = 1.
                        child->mixture_hist.back() = cache[i][assign[i]].mix->clone();
                        child->existence_probability = 1.0;
                        child->meas_assoc_hist.push_back(assign[i]);
                    } else {
                        // Missed: keep mixture, update existence.
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

                new_hypotheses.push_back(std::move(new_hyp));
            }
        }
        this->hypotheses_ = std::move(new_hypotheses);
    }

    void cleanup() override {
        this->normalize_log_weights();
        this->prune_hypotheses();
        this->cap_hypotheses();
        this->prune_low_existence_tracks();
        this->compact_track_table();
        this->normalize_log_weights();
        this->compute_cardinality();
        this->extract_states();
        this->push_extracted_snapshot();
        ++this->time_index_cntr_;
    }

private:
    void correct_no_measurements() {
        for (auto& hyp : this->hypotheses_) {
            std::vector<std::size_t> new_indices;
            for (auto idx : hyp.track_indices) {
                const auto& orig = *this->track_tab_[idx];
                double r = orig.existence_probability;
                double r_new = r * (1.0 - this->prob_detection_)
                             / std::max(1.0 - r * this->prob_detection_,
                                        std::numeric_limits<double>::min());
                auto child = orig.clone();
                child->existence_probability = r_new;
                child->meas_assoc_hist.push_back(-1);
                std::size_t new_idx = this->track_tab_.size();
                this->track_tab_.push_back(std::move(child));
                new_indices.push_back(new_idx);
            }
            hyp.track_indices = std::move(new_indices);
        }
    }

    std::vector<std::pair<MixturePtr, double>> birth_terms_;
};

}  // namespace brew::ggiw_orientation
