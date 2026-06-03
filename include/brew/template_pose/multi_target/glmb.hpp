#pragma once

// GLMB RFS for the template_pose package (concrete; brew::template_pose).

#include "brew/template_pose/template_pose_model.hpp"

#include "brew/shared/rfs_base.hpp"
#include "brew/shared/rfs_detail.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/trajectory_window.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/filter_traits.hpp"
#include "brew/clustering/dbscan.hpp"
#include "brew/assignment/murty.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace brew::template_pose {

template <typename Scalar = double, int D = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class GLMB : public multi_target::RFSBase {
    using T = models::TemplatePose<Scalar, D>;
public:
    using Label = std::pair<int, int>;
    using MixturePtr = std::unique_ptr<models::Mixture<T, MaxComponents>>;

    struct TabEntry {
        Label label{};
        int time_index = 0;
        std::vector<MixturePtr> mixture_hist;
        std::vector<int> meas_assoc_hist;

        [[nodiscard]] std::unique_ptr<TabEntry> clone() const {
            auto r = std::make_unique<TabEntry>();
            r->label = label;
            r->time_index = time_index;
            r->mixture_hist.reserve(mixture_hist.size());
            for (const auto& m : mixture_hist) r->mixture_hist.push_back(m->clone());
            r->meas_assoc_hist = meas_assoc_hist;
            return r;
        }
    };

    struct Hypothesis {
        double assoc_prob = 0.0;
        std::vector<std::size_t> track_set;
        [[nodiscard]] std::size_t num_tracks() const { return track_set.size(); }
    };

    struct ExtractedTrack {
        Label label{};
        int b_time_index = 0;
        std::vector<int> meas_assoc_hist;
        std::vector<std::unique_ptr<T>> states;
    };

    GLMB() {
        Hypothesis h0;
        h0.assoc_prob = 1.0;
        hypotheses_.push_back(std::move(h0));
        cardinality_pmf_ = Eigen::VectorXd::Zero(1);
        cardinality_pmf_(0) = 1.0;
    }

    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto c = std::make_unique<GLMB<Scalar, D, MaxComponents>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        c->has_filter_ = has_filter_;
        for (const auto& t : track_tab_) c->track_tab_.push_back(t->clone());
        c->hypotheses_ = hypotheses_;
        for (const auto& bt : birth_terms_) {
            c->birth_terms_.emplace_back(bt.first->clone(), bt.second);
        }
        c->req_births_ = req_births_;
        c->req_surv_ = req_surv_;
        c->req_upd_ = req_upd_;
        c->prune_threshold_ = prune_threshold_;
        c->max_hypotheses_ = max_hypotheses_;
        c->extract_threshold_ = extract_threshold_;
        c->gate_threshold_ = gate_threshold_;
        c->gating_on_ = gating_on_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        c->cardinality_pmf_ = cardinality_pmf_;
        c->estimated_cardinality_ = estimated_cardinality_;
        c->time_index_cntr_ = time_index_cntr_;
        for (const auto& e : extractable_hists_) {
            auto ne = std::make_unique<ExtractedTrack>();
            ne->label = e->label;
            ne->b_time_index = e->b_time_index;
            ne->meas_assoc_hist = e->meas_assoc_hist;
            ne->states.reserve(e->states.size());
            for (const auto& s : e->states) ne->states.push_back(s->clone_typed());
            c->extractable_hists_.push_back(std::move(ne));
        }
        return c;
    }

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
        has_filter_ = true;
    }

    void set_birth_terms(
        std::vector<std::pair<MixturePtr, double>> terms)
    {
        birth_terms_ = std::move(terms);
        if (!birth_terms_.empty() && !birth_terms_[0].first->empty()) {
            is_extended_ = birth_terms_[0].first->component(0).is_extended();
            if (is_extended_ && !cluster_obj_) {
                cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
    }

    void set_birth_model(MixturePtr birth_mix, double p_birth) {
        birth_terms_.clear();
        if (!birth_mix || birth_mix->empty()) return;
        birth_terms_.emplace_back(std::move(birth_mix), p_birth);
        is_extended_ = birth_terms_[0].first->component(0).is_extended();
        if (is_extended_ && !cluster_obj_) {
            cluster_obj_ = std::make_shared<clustering::DBSCAN>();
        }
    }

    void set_req_births(int n) { req_births_ = n; }
    void set_req_surv(int n) { req_surv_ = n; }
    void set_req_upd(int n) { req_upd_ = n; }

    void set_k_best(int k) { req_births_ = req_surv_ = req_upd_ = k; }
    void set_prune_threshold(double t) { prune_threshold_ = t; }

    void set_prune_threshold_hypothesis(double t) { prune_threshold_ = t; }
    void set_max_hypotheses(int n) { max_hypotheses_ = n; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_gating_on(bool on) { gating_on_ = on; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::ClusterBase> obj) {
        cluster_obj_ = std::move(obj);
    }

    [[nodiscard]] const std::vector<std::unique_ptr<TabEntry>>& track_tab() const {
        return track_tab_;
    }
    [[nodiscard]] const std::vector<Hypothesis>& hypotheses() const {
        return hypotheses_;
    }
    [[nodiscard]] const std::vector<std::unique_ptr<ExtractedTrack>>& extracted_tracks() const {
        return extractable_hists_;
    }

    [[nodiscard]] const std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>>&
    extracted_mixtures() const { return extracted_mixtures_; }
    [[nodiscard]] double estimated_cardinality() const { return estimated_cardinality_; }
    [[nodiscard]] const Eigen::VectorXd& cardinality() const { return cardinality_pmf_; }

    [[nodiscard]] std::map<int, std::vector<Eigen::VectorXd>> track_histories() const {
        std::map<int, std::vector<Eigen::VectorXd>> out;
        for (const auto& trk : extractable_hists_) {
            int id = label_to_id(trk->label);
            auto& vec = out[id];
            vec.reserve(trk->states.size());
            for (const auto& s : trk->states) {
                vec.push_back(multi_target::detail::get_state_vector(*s));
            }
        }
        return out;
    }

    void predict(int , double dt) override {
        if (!has_filter_) return;

        std::vector<std::unique_ptr<TabEntry>> birth_tab;
        Eigen::VectorXd birth_log_cost(static_cast<int>(birth_terms_.size()));
        for (std::size_t i = 0; i < birth_terms_.size(); ++i) {
            const auto& [mix, p_b] = birth_terms_[i];
            double cost = -std::log(p_b / (1.0 - p_b));
            birth_log_cost(static_cast<int>(i)) = cost;
            auto entry = std::make_unique<TabEntry>();
            entry->label = {time_index_cntr_, static_cast<int>(i)};
            entry->time_index = time_index_cntr_;
            entry->mixture_hist.push_back(mix->clone());
            birth_tab.push_back(std::move(entry));
        }

        auto birth_paths = multi_target::detail::k_shortest_subsets(birth_log_cost, req_births_);

        double tot_b_prob_log = 0.0;
        for (const auto& [mix, p_b] : birth_terms_) {
            tot_b_prob_log += std::log(std::max(1.0 - p_b, std::numeric_limits<double>::min()));
        }

        std::vector<std::pair<double, std::vector<int>>> birth_hyps;
        birth_hyps.reserve(birth_paths.size());
        std::vector<double> birth_logs;
        birth_logs.reserve(birth_paths.size());
        for (const auto& sol : birth_paths) {
            double lw = tot_b_prob_log - sol.cost;
            birth_hyps.emplace_back(lw, sol.indices);
            birth_logs.push_back(lw);
        }
        double birth_lse = multi_target::detail::log_sum_exp(birth_logs);
        for (auto& [lw, ind] : birth_hyps) {
            lw = std::exp(lw - birth_lse);
        }

        std::vector<std::unique_ptr<TabEntry>> surv_tab;
        surv_tab.reserve(track_tab_.size());
        for (const auto& trk : track_tab_) {
            auto nt = trk->clone();
            const auto& last = *nt->mixture_hist.back();
            auto predicted = last.clone();
            filter_->predict_batch_dynamic(dt, *predicted);
            nt->mixture_hist.push_back(std::move(predicted));
            surv_tab.push_back(std::move(nt));
        }

        double sum_sqrt_w = 0.0;
        for (const auto& h : hypotheses_) sum_sqrt_w += std::sqrt(h.assoc_prob);

        std::vector<std::pair<double, std::vector<std::size_t>>> surv_hyps;
        std::vector<double> surv_logs;
        for (const auto& hyp : hypotheses_) {
            if (hyp.num_tracks() == 0) {
                double lw = std::log(std::max(hyp.assoc_prob, std::numeric_limits<double>::min()));
                surv_hyps.emplace_back(lw, std::vector<std::size_t>{});
                surv_logs.push_back(lw);
                continue;
            }
            const int n = static_cast<int>(hyp.num_tracks());
            Eigen::VectorXd log_cost(n);
            double pdeath_log = 0.0;
            for (int i = 0; i < n; ++i) {
                double ps = prob_survive_;
                double pd = 1.0 - ps;
                log_cost(i) = -std::log(ps / std::max(pd, std::numeric_limits<double>::min()));
                pdeath_log += std::log(std::max(pd, std::numeric_limits<double>::min()));
            }
            int k_req = static_cast<int>(std::round(
                req_surv_ * std::sqrt(hyp.assoc_prob) / std::max(sum_sqrt_w, std::numeric_limits<double>::min())));
            if (k_req < 1) k_req = 1;
            auto paths = multi_target::detail::k_shortest_subsets(log_cost, k_req);
            for (const auto& sol : paths) {
                double lw = pdeath_log + std::log(std::max(hyp.assoc_prob, std::numeric_limits<double>::min())) - sol.cost;
                std::vector<std::size_t> ts;
                ts.reserve(sol.indices.size());
                for (int idx : sol.indices) ts.push_back(hyp.track_set[idx]);
                surv_hyps.emplace_back(lw, std::move(ts));
                surv_logs.push_back(lw);
            }
        }
        double surv_lse = multi_target::detail::log_sum_exp(surv_logs);
        for (auto& [lw, ts] : surv_hyps) {
            lw = std::exp(lw - surv_lse);
        }

        const std::size_t n_birth_tracks = birth_tab.size();
        std::vector<Hypothesis> new_hyps;
        new_hyps.reserve(birth_hyps.size() * surv_hyps.size());
        double tot_w = 0.0;
        for (const auto& [b_prob, b_idx] : birth_hyps) {
            for (const auto& [s_prob, s_ts] : surv_hyps) {
                Hypothesis h;
                h.assoc_prob = b_prob * s_prob;
                tot_w += h.assoc_prob;
                h.track_set.reserve(b_idx.size() + s_ts.size());
                for (int bi : b_idx) h.track_set.push_back(static_cast<std::size_t>(bi));
                for (std::size_t si : s_ts) h.track_set.push_back(si + n_birth_tracks);
                new_hyps.push_back(std::move(h));
            }
        }
        if (tot_w > 0.0) {
            for (auto& h : new_hyps) h.assoc_prob /= tot_w;
        }

        track_tab_.clear();
        track_tab_.reserve(n_birth_tracks + surv_tab.size());
        for (auto& e : birth_tab) track_tab_.push_back(std::move(e));
        for (auto& e : surv_tab) track_tab_.push_back(std::move(e));

        hypotheses_ = std::move(new_hyps);
        clean_predictions();
        card_dist_from_hyps();
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!has_filter_) return;

        std::vector<Eigen::MatrixXd> meas_groups;
        if (is_extended_ && cluster_obj_) {
            meas_groups = cluster_obj_->cluster(measurements);
        } else {
            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }
        const int num_meas = static_cast<int>(meas_groups.size());
        const int num_pred = static_cast<int>(track_tab_.size());

        std::vector<std::unique_ptr<TabEntry>> cor_tab;
        cor_tab.resize(static_cast<std::size_t>((num_meas + 1) * num_pred));

        for (int i = 0; i < num_pred; ++i) {
            auto nt = track_tab_[i]->clone();
            nt->meas_assoc_hist.push_back(-1);
            cor_tab[static_cast<std::size_t>(i)] = std::move(nt);
        }

        Eigen::MatrixXd all_cost_m = Eigen::MatrixXd::Zero(num_pred, std::max(num_meas, 1));

        for (int j = 0; j < num_meas; ++j) {
            const Eigen::MatrixXd& meas = meas_groups[j];
            Eigen::VectorXd z_gate = (is_extended_ && meas.cols() > 1)
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
                const auto& trk = *track_tab_[i];
                const auto& last = *trk.mixture_hist.back();

                if (gating_on_) {

                    bool gated = false;
                    for (std::size_t c = 0; c < last.size(); ++c) {
                        double gv = filter_->gate(z_gate, last.component(c));
                        if (gv < gate_threshold_) { gated = true; break; }
                    }
                    if (!gated) {
                        all_cost_m(i, j) = 0.0;
                        continue;
                    }
                }

                auto corrected = std::make_unique<models::Mixture<T, MaxComponents>>();
                std::vector<double> new_w(last.size(), 0.0);
                double total_w = 0.0;
                for (std::size_t c = 0; c < last.size(); ++c) {
                    auto [dist, qz] = filter_->correct(meas_flat, last.component(c));
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

                auto nt = track_tab_[i]->clone();
                nt->mixture_hist.back() = std::move(corrected);
                nt->meas_assoc_hist.push_back(j);
                std::size_t slot = static_cast<std::size_t>(num_pred) * (j + 1)
                                 + static_cast<std::size_t>(i);
                cor_tab[slot] = std::move(nt);
            }
        }

        std::vector<Hypothesis> up_hyps;

        if (num_meas == 0) {
            double pmd_single = std::log(std::max(1.0 - prob_detection_,
                                                  std::numeric_limits<double>::min()));
            for (const auto& hyp : hypotheses_) {
                double pmd_log = hyp.track_set.size() * pmd_single;
                double lw = -clutter_rate_ + pmd_log
                          + std::log(std::max(hyp.assoc_prob, std::numeric_limits<double>::min()));
                Hypothesis h;
                h.assoc_prob = lw;
                h.track_set = hyp.track_set;
                up_hyps.push_back(std::move(h));
            }
        } else {
            const double kappa_base = clutter_rate_ * clutter_density_;
            std::vector<double> kappa_vec(num_meas);
            double total_clutter_log = 0.0;
            for (int j = 0; j < num_meas; ++j) {
                const int W = static_cast<int>(meas_groups[j].cols());
                kappa_vec[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
                total_clutter_log += std::log(std::max(kappa_vec[j],
                                                       std::numeric_limits<double>::min()));
            }
            double ss_w = 0.0;
            for (const auto& p : hypotheses_) ss_w += std::sqrt(p.assoc_prob);
            ss_w = std::max(ss_w, std::numeric_limits<double>::min());

            for (const auto& p_hyp : hypotheses_) {
                if (p_hyp.num_tracks() == 0) {
                    double lw = -clutter_rate_ + total_clutter_log
                              + std::log(std::max(p_hyp.assoc_prob,
                                                  std::numeric_limits<double>::min()));
                    Hypothesis h;
                    h.assoc_prob = lw;
                    h.track_set = p_hyp.track_set;
                    up_hyps.push_back(std::move(h));
                    continue;
                }

                const int nt = static_cast<int>(p_hyp.num_tracks());
                const int ncols = num_meas + nt;
                Eigen::MatrixXd neg_log = Eigen::MatrixXd::Constant(nt, ncols,
                    std::numeric_limits<double>::infinity());

                double pmd_log = 0.0;
                for (int i = 0; i < nt; ++i) {
                    double pd = prob_detection_;
                    double pmd = std::max(1.0 - pd, std::numeric_limits<double>::min());
                    pmd_log += std::log(pmd);

                    double ratio_ = pd / pmd;
                    for (int j = 0; j < num_meas; ++j) {
                        double like = all_cost_m(static_cast<Eigen::Index>(p_hyp.track_set[i]), j);
                        double c = ratio_ * like
                                 / std::max(kappa_vec[j], std::numeric_limits<double>::min());
                        if (c > 0.0 && std::isfinite(c)) {
                            neg_log(i, j) = -std::log(c);
                        }
                    }
                    neg_log(i, num_meas + i) = 0.0;
                }

                int m_req = static_cast<int>(std::round(
                    req_upd_ * std::sqrt(p_hyp.assoc_prob) / ss_w));
                if (m_req < 1) m_req = 1;

                auto solutions = assignment::murty(neg_log, m_req);

                for (const auto& sol : solutions) {
                    std::vector<int> assigned(nt, -1);
                    for (const auto& [row, col] : sol.assignments) {
                        if (row < nt && col < num_meas) assigned[row] = col;
                    }
                    Hypothesis h;
                    double lw = -clutter_rate_ + total_clutter_log + pmd_log
                              + std::log(std::max(p_hyp.assoc_prob,
                                                  std::numeric_limits<double>::min()))
                              - sol.total_cost;
                    h.assoc_prob = lw;
                    h.track_set.reserve(nt);
                    for (int i = 0; i < nt; ++i) {
                        std::size_t orig = p_hyp.track_set[i];
                        std::size_t slot = (assigned[i] >= 0)
                            ? static_cast<std::size_t>(num_pred) * (assigned[i] + 1) + orig
                            : orig;
                        h.track_set.push_back(slot);
                    }
                    up_hyps.push_back(std::move(h));
                }
            }
        }

        std::vector<double> logs;
        logs.reserve(up_hyps.size());
        for (const auto& h : up_hyps) logs.push_back(h.assoc_prob);
        double lse = multi_target::detail::log_sum_exp(logs);
        for (auto& h : up_hyps) h.assoc_prob = std::exp(h.assoc_prob - lse);

        track_tab_ = std::move(cor_tab);
        hypotheses_ = std::move(up_hyps);
        card_dist_from_hyps();
        clean_updates();
    }

    void cleanup() override {
        prune_hyps();
        cap_hyps();
        compact_track_tab();
        card_dist_from_hyps();
        extract_states();
        push_extracted_snapshot();
        ++time_index_cntr_;
    }

protected:

    void clean_predictions() {
        std::vector<Hypothesis> out;
        std::vector<std::vector<std::size_t>> keys;
        for (auto& h : hypotheses_) {
            auto key = h.track_set;
            std::sort(key.begin(), key.end());
            bool merged = false;
            for (std::size_t k = 0; k < keys.size(); ++k) {
                if (keys[k] == key) {
                    out[k].assoc_prob += h.assoc_prob;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                keys.push_back(std::move(key));
                out.push_back(std::move(h));
            }
        }
        hypotheses_ = std::move(out);
    }

    void clean_updates() {
        const std::size_t n = track_tab_.size();
        std::vector<std::size_t> used(n, 0);
        for (const auto& h : hypotheses_) {
            for (auto idx : h.track_set) {
                if (idx < n && track_tab_[idx]) ++used[idx];
            }
        }
        std::vector<std::size_t> new_idx(n, std::numeric_limits<std::size_t>::max());
        std::size_t next = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i] > 0) new_idx[i] = next++;
        }
        std::vector<std::unique_ptr<TabEntry>> compact;
        compact.reserve(next);
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i] > 0) compact.push_back(std::move(track_tab_[i]));
        }
        std::vector<Hypothesis> kept;
        kept.reserve(hypotheses_.size());
        for (auto& h : hypotheses_) {
            bool ok = true;
            std::vector<std::size_t> ts;
            ts.reserve(h.track_set.size());
            for (auto idx : h.track_set) {
                if (idx >= n || new_idx[idx] == std::numeric_limits<std::size_t>::max()) {
                    ok = false; break;
                }
                ts.push_back(new_idx[idx]);
            }
            if (ok) {
                h.track_set = std::move(ts);
                kept.push_back(std::move(h));
            }
        }
        track_tab_ = std::move(compact);
        hypotheses_ = std::move(kept);
    }

    void compact_track_tab() { clean_updates(); }

    void prune_hyps() {
        std::vector<Hypothesis> kept;
        double kept_sum = 0.0;
        for (auto& h : hypotheses_) {
            if (h.assoc_prob > prune_threshold_) {
                kept_sum += h.assoc_prob;
                kept.push_back(std::move(h));
            }
        }
        if (kept.empty() && !hypotheses_.empty()) {
            auto best = std::max_element(hypotheses_.begin(), hypotheses_.end(),
                [](const auto& a, const auto& b) { return a.assoc_prob < b.assoc_prob; });
            kept_sum = best->assoc_prob;
            kept.push_back(std::move(*best));
        }
        if (kept_sum > 0.0) {
            for (auto& h : kept) h.assoc_prob /= kept_sum;
        }
        hypotheses_ = std::move(kept);
    }

    void cap_hyps() {
        if (static_cast<int>(hypotheses_.size()) <= max_hypotheses_) return;
        std::sort(hypotheses_.begin(), hypotheses_.end(),
            [](const auto& a, const auto& b) { return a.assoc_prob > b.assoc_prob; });
        hypotheses_.resize(static_cast<std::size_t>(max_hypotheses_));
        double s = 0.0;
        for (const auto& h : hypotheses_) s += h.assoc_prob;
        if (s > 0.0) for (auto& h : hypotheses_) h.assoc_prob /= s;
    }

    void card_dist_from_hyps() {
        if (hypotheses_.empty()) {
            cardinality_pmf_ = Eigen::VectorXd::Zero(1);
            estimated_cardinality_ = 0.0;
            return;
        }
        std::size_t max_card = 0;
        for (const auto& h : hypotheses_) max_card = std::max(max_card, h.num_tracks());
        cardinality_pmf_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(max_card + 1));
        for (const auto& h : hypotheses_) {
            cardinality_pmf_(static_cast<Eigen::Index>(h.num_tracks())) += h.assoc_prob;
        }
        double s = cardinality_pmf_.sum();
        if (s > 0.0) cardinality_pmf_ /= s;
        estimated_cardinality_ = 0.0;
        for (int n = 0; n < cardinality_pmf_.size(); ++n) {
            estimated_cardinality_ += n * cardinality_pmf_(n);
        }
    }

    std::vector<std::unique_ptr<T>> extract_helper(const TabEntry& trk) const {
        std::vector<std::unique_ptr<T>> states;
        states.reserve(trk.mixture_hist.size());
        for (const auto& mix : trk.mixture_hist) {
            std::size_t best = 0;
            double best_w = -1.0;
            for (std::size_t c = 0; c < mix->size(); ++c) {
                double w = mix->weight(c);
                if (w > best_w) { best_w = w; best = c; }
            }
            states.push_back(mix->component(best).clone_typed());
        }
        return states;
    }

    void update_extract_hist(std::size_t idx_cmp) {
        std::vector<std::vector<int>> used_meas(static_cast<std::size_t>(time_index_cntr_ + 1));
        std::vector<Label> used_labels;
        std::vector<std::unique_ptr<ExtractedTrack>> new_hists;
        for (auto tab_idx : hypotheses_[idx_cmp].track_set) {
            const auto& trk = *track_tab_[tab_idx];
            auto e = std::make_unique<ExtractedTrack>();
            e->label = trk.label;
            e->meas_assoc_hist = trk.meas_assoc_hist;
            e->b_time_index = trk.time_index;
            e->states = extract_helper(trk);
            used_labels.push_back(trk.label);
            for (std::size_t t = 0; t < e->meas_assoc_hist.size(); ++t) {
                int tt = e->b_time_index + static_cast<int>(t);
                int m = e->meas_assoc_hist[t];
                if (tt >= 0 && static_cast<std::size_t>(tt) < used_meas.size() && m >= 0) {
                    used_meas[static_cast<std::size_t>(tt)].push_back(m);
                }
            }
            new_hists.push_back(std::move(e));
        }
        std::vector<std::unique_ptr<ExtractedTrack>> surviving;
        for (auto& existing : extractable_hists_) {
            bool claimed = false;
            for (const auto& l : used_labels) {
                if (l == existing->label) { claimed = true; break; }
            }
            if (claimed) continue;
            for (std::size_t t = 0; t < existing->meas_assoc_hist.size(); ++t) {
                int tt = existing->b_time_index + static_cast<int>(t);
                int m = existing->meas_assoc_hist[t];
                if (m < 0) continue;
                if (tt < 0 || static_cast<std::size_t>(tt) >= used_meas.size()) continue;
                for (int um : used_meas[static_cast<std::size_t>(tt)]) {
                    if (um == m) { claimed = true; break; }
                }
                if (claimed) break;
            }
            if (!claimed) surviving.push_back(std::move(existing));
        }
        for (auto& e : new_hists) surviving.push_back(std::move(e));
        extractable_hists_ = std::move(surviving);
    }

    [[nodiscard]] std::size_t map_hypothesis_index() const {
        if (hypotheses_.empty()) return std::numeric_limits<std::size_t>::max();
        int map_card = 0;
        double best_p = -1.0;
        for (int n = 0; n < cardinality_pmf_.size(); ++n) {
            if (cardinality_pmf_(n) > best_p) { best_p = cardinality_pmf_(n); map_card = n; }
        }
        std::size_t idx = std::numeric_limits<std::size_t>::max();
        double best_w = -1.0;
        for (std::size_t i = 0; i < hypotheses_.size(); ++i) {
            if (static_cast<int>(hypotheses_[i].num_tracks()) == map_card
                && hypotheses_[i].assoc_prob > best_w) {
                best_w = hypotheses_[i].assoc_prob;
                idx = i;
            }
        }
        return idx;
    }

    void extract_states() {
        std::size_t idx_cmp = map_hypothesis_index();
        if (idx_cmp == std::numeric_limits<std::size_t>::max()) return;
        update_extract_hist(idx_cmp);
    }

    static int label_to_id(const Label& l) {

        return l.first * 100000 + l.second;
    }

    void push_extracted_snapshot() {
        if (this->max_history_ == 0) return;
        auto mix = std::make_unique<models::Mixture<T, MaxComponents>>();
        std::size_t idx_cmp = map_hypothesis_index();
        if (idx_cmp != std::numeric_limits<std::size_t>::max()) {
            for (auto tab_idx : hypotheses_[idx_cmp].track_set) {
                if (tab_idx >= track_tab_.size()) continue;
                const auto& trk = *track_tab_[tab_idx];
                if (trk.mixture_hist.empty()) continue;
                const auto& last_mix = *trk.mixture_hist.back();
                if (last_mix.empty()) continue;
                std::size_t best = 0;
                double best_w = -1.0;
                for (std::size_t c = 0; c < last_mix.size(); ++c) {
                    if (last_mix.weight(c) > best_w) { best_w = last_mix.weight(c); best = c; }
                }
                mix->add_component(last_mix.component(best).clone_typed(), 1.0);
            }
        }
        this->push_history(extracted_mixtures_, std::move(mix));
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    bool has_filter_ = false;
    std::vector<std::unique_ptr<TabEntry>> track_tab_;
    std::vector<Hypothesis> hypotheses_;
    std::vector<std::unique_ptr<ExtractedTrack>> extractable_hists_;
    std::vector<std::pair<MixturePtr, double>> birth_terms_;

    int req_births_ = 5;
    int req_surv_ = 5;
    int req_upd_ = 5;
    double prune_threshold_ = 1e-15;
    int max_hypotheses_ = 3000;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    bool gating_on_ = false;
    bool is_extended_ = false;
    std::shared_ptr<clustering::ClusterBase> cluster_obj_;

    Eigen::VectorXd cardinality_pmf_;
    double estimated_cardinality_ = 0.0;
    int time_index_cntr_ = 0;
    std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>> extracted_mixtures_;
};

}
