#pragma once

#include "brew/shared/multi_target_generic/glmb.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace brew::multi_target {

/// Joint GLMB filter (Vo, Vo & Phung 2017). Inherits GLMB's track-table / hypothesis
/// machinery but moves birth from predict() into the correct() assignment: birth
/// candidates compete directly with surviving tracks for measurements.
// @mex rfs
// @mex_name JGLMB
// @mex_params req_surv:int:5, req_upd:int:5, prune_threshold:double:1e-15, max_hypotheses:int:3000, extract_threshold:double:0.5, gate_threshold:double:9.0, gating_on:bool:false
// @mex_has cardinality, track_histories, cluster_object
template <typename T, int MaxComponents = Eigen::Dynamic>
class JGLMB : public GLMB<T, MaxComponents> {
public:
    using Base = GLMB<T, MaxComponents>;
    using Label = typename Base::Label;
    using MixturePtr = typename Base::MixturePtr;
    using TabEntry = typename Base::TabEntry;
    using Hypothesis = typename Base::Hypothesis;
    using ExtractedTrack = typename Base::ExtractedTrack;

    JGLMB() : Base() {}

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<JGLMB<T, MaxComponents>>();
        c->prob_detection_ = this->prob_detection_;
        c->prob_survive_ = this->prob_survive_;
        c->clutter_rate_ = this->clutter_rate_;
        c->clutter_density_ = this->clutter_density_;
        c->filter_ = this->filter_;
        c->has_filter_ = this->has_filter_;
        for (const auto& t : this->track_tab_) c->track_tab_.push_back(t->clone());
        c->hypotheses_ = this->hypotheses_;
        for (const auto& bt : this->birth_terms_) {
            c->birth_terms_.emplace_back(bt.first->clone(), bt.second);
        }
        c->req_births_ = this->req_births_;
        c->req_surv_ = this->req_surv_;
        c->req_upd_ = this->req_upd_;
        c->prune_threshold_ = this->prune_threshold_;
        c->max_hypotheses_ = this->max_hypotheses_;
        c->extract_threshold_ = this->extract_threshold_;
        c->gate_threshold_ = this->gate_threshold_;
        c->gating_on_ = this->gating_on_;
        c->is_extended_ = this->is_extended_;
        if (this->cluster_obj_) c->cluster_obj_ = this->cluster_obj_;
        c->cardinality_pmf_ = this->cardinality_pmf_;
        c->estimated_cardinality_ = this->estimated_cardinality_;
        c->time_index_cntr_ = this->time_index_cntr_;
        for (const auto& e : this->extractable_hists_) {
            auto ne = std::make_unique<ExtractedTrack>();
            ne->label = e->label;
            ne->b_time_index = e->b_time_index;
            ne->meas_assoc_hist = e->meas_assoc_hist;
            ne->states.reserve(e->states.size());
            for (const auto& s : e->states) ne->states.push_back(s->clone_typed());
            c->extractable_hists_.push_back(std::move(ne));
        }
        for (const auto& m : this->extracted_mixtures_) c->extracted_mixtures_.push_back(m->clone());
        return c;
    }

    /// Predict existing tracks only (no separate birth step).
    void predict(int /*timestep*/, double dt) override {
        if (!this->has_filter_) return;

        std::vector<std::unique_ptr<TabEntry>> surv_tab;
        surv_tab.reserve(this->track_tab_.size());
        for (const auto& trk : this->track_tab_) {
            auto nt = trk->clone();
            const auto& last = *nt->mixture_hist.back();
            auto predicted = last.clone();
            this->filter_.predict_batch(dt, *predicted);
            nt->mixture_hist.push_back(std::move(predicted));
            surv_tab.push_back(std::move(nt));
        }

        double sum_sqrt_w = 0.0;
        for (const auto& h : this->hypotheses_) sum_sqrt_w += std::sqrt(h.assoc_prob);

        std::vector<Hypothesis> new_hyps;
        std::vector<double> logs;
        for (const auto& hyp : this->hypotheses_) {
            if (hyp.num_tracks() == 0) {
                double lw = std::log(std::max(hyp.assoc_prob,
                                              std::numeric_limits<double>::min()));
                Hypothesis h;
                h.assoc_prob = lw;
                new_hyps.push_back(std::move(h));
                logs.push_back(lw);
                continue;
            }
            const int n = static_cast<int>(hyp.num_tracks());
            Eigen::VectorXd log_cost(n);
            double pdeath_log = 0.0;
            for (int i = 0; i < n; ++i) {
                double ps = this->prob_survive_;
                double pd = 1.0 - ps;
                log_cost(i) = -std::log(ps / std::max(pd, std::numeric_limits<double>::min()));
                pdeath_log += std::log(std::max(pd, std::numeric_limits<double>::min()));
            }
            int k_req = static_cast<int>(std::round(
                this->req_surv_ * std::sqrt(hyp.assoc_prob)
                / std::max(sum_sqrt_w, std::numeric_limits<double>::min())));
            if (k_req < 1) k_req = 1;
            auto paths = detail::k_shortest_subsets(log_cost, k_req);
            for (const auto& sol : paths) {
                double lw = pdeath_log
                          + std::log(std::max(hyp.assoc_prob,
                                              std::numeric_limits<double>::min()))
                          - sol.cost;
                Hypothesis h;
                h.assoc_prob = lw;
                h.track_set.reserve(sol.indices.size());
                for (int idx : sol.indices) h.track_set.push_back(hyp.track_set[idx]);
                new_hyps.push_back(std::move(h));
                logs.push_back(lw);
            }
        }
        double lse = detail::log_sum_exp(logs);
        for (auto& h : new_hyps) h.assoc_prob = std::exp(h.assoc_prob - lse);

        this->track_tab_ = std::move(surv_tab);
        this->hypotheses_ = std::move(new_hyps);
        this->clean_predictions();
        this->card_dist_from_hyps();
    }

    /// Joint correct: birth candidates compete with surviving tracks for measurements
    /// within the same Murty assignment.
    void correct(const Eigen::MatrixXd& measurements) override {
        if (!this->has_filter_) return;

        std::vector<Eigen::MatrixXd> meas_groups;
        if (this->is_extended_ && this->cluster_obj_) {
            meas_groups = this->cluster_obj_->cluster(measurements);
        } else {
            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }
        const int num_meas = static_cast<int>(meas_groups.size());
        const int num_pred = static_cast<int>(this->track_tab_.size());
        const int num_birth = static_cast<int>(this->birth_terms_.size());

        const std::size_t birth_base = static_cast<std::size_t>(num_pred) * (num_meas + 1);
        std::vector<std::unique_ptr<TabEntry>> cor_tab;
        cor_tab.resize(birth_base + static_cast<std::size_t>(num_birth) * num_meas);

        for (int i = 0; i < num_pred; ++i) {
            auto nt = this->track_tab_[i]->clone();
            nt->meas_assoc_hist.push_back(-1);
            cor_tab[static_cast<std::size_t>(i)] = std::move(nt);
        }

        Eigen::MatrixXd all_cost_m = Eigen::MatrixXd::Zero(num_pred, std::max(num_meas, 1));
        Eigen::MatrixXd birth_cost_m = Eigen::MatrixXd::Zero(num_birth, std::max(num_meas, 1));

        for (int j = 0; j < num_meas; ++j) {
            const Eigen::MatrixXd& meas = meas_groups[j];
            Eigen::VectorXd z_gate = (this->is_extended_ && meas.cols() > 1)
                ? Eigen::VectorXd(meas.rowwise().mean())
                : Eigen::VectorXd(meas.col(0));
            Eigen::VectorXd meas_flat;
            if (meas.cols() > 1) {
                meas_flat.resize(meas.size());
                for (int c = 0; c < meas.cols(); ++c) {
                    meas_flat.segment(c * meas.rows(), meas.rows()) = meas.col(c);
                }
            } else {
                meas_flat = meas.col(0);
            }

            for (int i = 0; i < num_pred; ++i) {
                const auto& trk = *this->track_tab_[i];
                const auto& last = *trk.mixture_hist.back();
                if (this->gating_on_) {
                    bool gated = false;
                    for (std::size_t c = 0; c < last.size(); ++c) {
                        if (this->filter_.gate(z_gate, last.component(c)) < this->gate_threshold_) {
                            gated = true; break;
                        }
                    }
                    if (!gated) { all_cost_m(i, j) = 0.0; continue; }
                }
                auto corrected = std::make_unique<models::Mixture<T, MaxComponents>>();
                std::vector<double> new_w(last.size(), 0.0);
                double total_w = 0.0;
                for (std::size_t c = 0; c < last.size(); ++c) {
                    auto [dist, qz] = this->filter_.correct(meas_flat, last.component(c));
                    double w = last.weight(c) * qz;
                    new_w[c] = w;
                    total_w += w;
                    corrected->add_component(dist.clone_typed(), w);
                }
                total_w += std::numeric_limits<double>::epsilon();
                for (std::size_t c = 0; c < corrected->size(); ++c) {
                    corrected->weights()(static_cast<Eigen::Index>(c)) = new_w[c] / total_w;
                }
                all_cost_m(i, j) = total_w;

                auto nt = this->track_tab_[i]->clone();
                nt->mixture_hist.back() = std::move(corrected);
                nt->meas_assoc_hist.push_back(j);
                std::size_t slot = static_cast<std::size_t>(num_pred) * (j + 1)
                                 + static_cast<std::size_t>(i);
                cor_tab[slot] = std::move(nt);
            }

            for (int b = 0; b < num_birth; ++b) {
                const auto& [birth_mix, p_b] = this->birth_terms_[b];
                auto corrected = std::make_unique<models::Mixture<T, MaxComponents>>();
                std::vector<double> new_w(birth_mix->size(), 0.0);
                double total_w = 0.0;
                for (std::size_t c = 0; c < birth_mix->size(); ++c) {
                    auto [dist, qz] = this->filter_.correct(meas_flat, birth_mix->component(c));
                    double w = birth_mix->weight(c) * qz;
                    new_w[c] = w;
                    total_w += w;
                    corrected->add_component(dist.clone_typed(), w);
                }
                total_w += std::numeric_limits<double>::epsilon();
                for (std::size_t c = 0; c < corrected->size(); ++c) {
                    corrected->weights()(static_cast<Eigen::Index>(c)) = new_w[c] / total_w;
                }
                birth_cost_m(b, j) = total_w;

                auto nt = std::make_unique<TabEntry>();
                nt->label = {this->time_index_cntr_, b};
                nt->time_index = this->time_index_cntr_;
                nt->mixture_hist.push_back(std::move(corrected));
                nt->meas_assoc_hist.push_back(j);
                std::size_t slot = birth_base
                                 + static_cast<std::size_t>(num_birth) * j
                                 + static_cast<std::size_t>(b);
                cor_tab[slot] = std::move(nt);
            }
        }

        std::vector<Hypothesis> up_hyps;

        if (num_meas == 0) {
            double pmd = std::max(1.0 - this->prob_detection_,
                                  std::numeric_limits<double>::min());
            double no_birth_log = 0.0;
            for (const auto& [mix, p_b] : this->birth_terms_) {
                no_birth_log += std::log(std::max(1.0 - p_b,
                                                  std::numeric_limits<double>::min()));
            }
            for (const auto& hyp : this->hypotheses_) {
                double pmd_log = hyp.track_set.size() * std::log(pmd);
                double lw = -this->clutter_rate_ + pmd_log + no_birth_log
                          + std::log(std::max(hyp.assoc_prob,
                                              std::numeric_limits<double>::min()));
                Hypothesis h;
                h.assoc_prob = lw;
                h.track_set = hyp.track_set;
                up_hyps.push_back(std::move(h));
            }
        } else {
            const double kappa_base = this->clutter_rate_ * this->clutter_density_;
            std::vector<double> kappa_vec(num_meas);
            double total_clutter_log = 0.0;
            for (int j = 0; j < num_meas; ++j) {
                const int W = static_cast<int>(meas_groups[j].cols());
                kappa_vec[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
                total_clutter_log += std::log(std::max(kappa_vec[j],
                                                       std::numeric_limits<double>::min()));
            }
            double ss_w = 0.0;
            for (const auto& p : this->hypotheses_) ss_w += std::sqrt(p.assoc_prob);
            ss_w = std::max(ss_w, std::numeric_limits<double>::min());

            double no_birth_log = 0.0;
            for (const auto& [mix, p_b] : this->birth_terms_) {
                no_birth_log += std::log(std::max(1.0 - p_b,
                                                  std::numeric_limits<double>::min()));
            }

            for (const auto& p_hyp : this->hypotheses_) {
                const int nt = static_cast<int>(p_hyp.num_tracks());
                const int n_rows = nt + num_birth;
                const int n_cols = num_meas + n_rows;
                Eigen::MatrixXd neg_log = Eigen::MatrixXd::Constant(n_rows, n_cols,
                    std::numeric_limits<double>::infinity());

                double pmd_log = 0.0;
                double pd = this->prob_detection_;
                double pmd = std::max(1.0 - pd, std::numeric_limits<double>::min());
                double ratio_ = pd / pmd;

                for (int i = 0; i < nt; ++i) {
                    pmd_log += std::log(pmd);
                    for (int j = 0; j < num_meas; ++j) {
                        double like = all_cost_m(
                            static_cast<Eigen::Index>(p_hyp.track_set[i]), j);
                        double c = ratio_ * like
                                 / std::max(kappa_vec[j], std::numeric_limits<double>::min());
                        if (c > 0.0 && std::isfinite(c)) {
                            neg_log(i, j) = -std::log(c);
                        }
                    }
                    neg_log(i, num_meas + i) = 0.0;
                }

                for (int b = 0; b < num_birth; ++b) {
                    int row = nt + b;
                    double p_b = this->birth_terms_[b].second;
                    double p_nb = std::max(1.0 - p_b,
                                           std::numeric_limits<double>::min());
                    double ratio_b = p_b / p_nb;
                    for (int j = 0; j < num_meas; ++j) {
                        double like = birth_cost_m(b, j);
                        double c = ratio_b * like
                                 / std::max(kappa_vec[j], std::numeric_limits<double>::min());
                        if (c > 0.0 && std::isfinite(c)) {
                            neg_log(row, j) = -std::log(c);
                        }
                    }
                    neg_log(row, num_meas + row) = 0.0;
                }

                int m_req = static_cast<int>(std::round(
                    this->req_upd_ * std::sqrt(p_hyp.assoc_prob) / ss_w));
                if (m_req < 1) m_req = 1;

                auto solutions = assignment::murty(neg_log, m_req);

                for (const auto& sol : solutions) {
                    std::vector<int> assigned(n_rows, -1);
                    for (const auto& [row, col] : sol.assignments) {
                        if (row < n_rows && col < num_meas) assigned[row] = col;
                    }
                    Hypothesis h;
                    double lw = -this->clutter_rate_ + total_clutter_log
                              + pmd_log + no_birth_log
                              + std::log(std::max(p_hyp.assoc_prob,
                                                  std::numeric_limits<double>::min()))
                              - sol.total_cost;
                    h.assoc_prob = lw;
                    h.track_set.reserve(n_rows);
                    for (int i = 0; i < nt; ++i) {
                        std::size_t orig = p_hyp.track_set[i];
                        std::size_t slot = (assigned[i] >= 0)
                            ? static_cast<std::size_t>(num_pred) * (assigned[i] + 1) + orig
                            : orig;
                        h.track_set.push_back(slot);
                    }
                    for (int b = 0; b < num_birth; ++b) {
                        int row = nt + b;
                        if (assigned[row] >= 0) {
                            std::size_t slot = birth_base
                                             + static_cast<std::size_t>(num_birth) * assigned[row]
                                             + static_cast<std::size_t>(b);
                            h.track_set.push_back(slot);
                        }
                    }
                    up_hyps.push_back(std::move(h));
                }
            }
        }

        std::vector<double> logs;
        logs.reserve(up_hyps.size());
        for (const auto& h : up_hyps) logs.push_back(h.assoc_prob);
        double lse = detail::log_sum_exp(logs);
        for (auto& h : up_hyps) h.assoc_prob = std::exp(h.assoc_prob - lse);

        this->track_tab_ = std::move(cor_tab);
        this->hypotheses_ = std::move(up_hyps);
        this->card_dist_from_hyps();
        this->clean_updates();
    }
};

}  // namespace brew::multi_target
