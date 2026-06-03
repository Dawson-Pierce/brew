#pragma once

// MBMBase RFS for the trajectory_gaussian package (concrete; brew::trajectory_gaussian).

#include "brew/trajectory_gaussian/trajectory_gaussian_model.hpp"

#include "brew/shared/rfs_base.hpp"
#include "brew/shared/rfs_detail.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/filter_traits.hpp"
#include "brew/clustering/dbscan.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace brew::trajectory_gaussian {

template <typename Scalar = double, int D = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class MBMBase : public multi_target::RFSBase {
    using T = models::TrajectoryGaussian<Scalar, D>;
public:
    using MixturePtr = std::unique_ptr<models::Mixture<T, MaxComponents>>;

    struct TrackEntry {
        int id = -1;
        double existence_probability = 0.0;
        int birth_time_index = 0;
        std::vector<MixturePtr> mixture_hist;
        std::vector<int> meas_assoc_hist;

        [[nodiscard]] std::unique_ptr<TrackEntry> clone() const {
            auto r = std::make_unique<TrackEntry>();
            r->id = id;
            r->existence_probability = existence_probability;
            r->birth_time_index = birth_time_index;
            r->mixture_hist.reserve(mixture_hist.size());
            for (const auto& m : mixture_hist) r->mixture_hist.push_back(m->clone());
            r->meas_assoc_hist = meas_assoc_hist;
            return r;
        }

        [[nodiscard]] const models::Mixture<T, MaxComponents>& current_mixture() const {
            return *mixture_hist.back();
        }
        [[nodiscard]] models::Mixture<T, MaxComponents>& current_mixture() {
            return *mixture_hist.back();
        }
    };

    struct Hypothesis {
        double log_weight = 0.0;
        std::vector<std::size_t> track_indices;
        [[nodiscard]] std::size_t num_tracks() const { return track_indices.size(); }
    };

    struct ExtractedTrack {
        int id = -1;
        int b_time_index = 0;
        std::vector<int> meas_assoc_hist;
        std::vector<std::unique_ptr<T>> states;
    };

    MBMBase() = default;

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
        has_filter_ = true;
    }
    void set_prune_threshold_hypothesis(double t) { prune_threshold_hyp_ = t; }
    void set_prune_threshold_bernoulli(double t) { prune_threshold_bern_ = t; }
    void set_max_hypotheses(int n) { max_hypotheses_ = n; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_gating_on(bool on) { gating_on_ = on; }
    void set_k_best(int k) { k_best_ = k; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::ClusterBase> obj) {
        cluster_obj_ = std::move(obj);
    }

    [[nodiscard]] const std::vector<std::unique_ptr<TrackEntry>>& track_tab() const {
        return track_tab_;
    }
    [[nodiscard]] const std::vector<Hypothesis>& hypotheses() const { return hypotheses_; }
    [[nodiscard]] const std::vector<std::unique_ptr<ExtractedTrack>>& extracted_tracks() const {
        return extracted_hists_;
    }
    [[nodiscard]] const std::deque<MixturePtr>& extracted_mixtures() const {
        return extracted_mixtures_;
    }
    [[nodiscard]] double estimated_cardinality() const { return estimated_cardinality_; }
    [[nodiscard]] const Eigen::VectorXd& cardinality() const { return cardinality_pmf_; }

    [[nodiscard]] std::map<int, std::vector<Eigen::VectorXd>> track_histories() const {
        std::map<int, std::vector<Eigen::VectorXd>> out;
        for (const auto& trk : extracted_hists_) {
            auto& vec = out[trk->id];
            vec.reserve(trk->states.size());
            for (const auto& s : trk->states) {
                vec.push_back(multi_target::detail::get_state_vector(*s));
            }
        }
        return out;
    }

protected:

    void normalize_log_weights() {
        if (hypotheses_.empty()) return;
        double max_lw = hypotheses_[0].log_weight;
        for (const auto& h : hypotheses_) max_lw = std::max(max_lw, h.log_weight);
        double sum_exp = 0.0;
        for (const auto& h : hypotheses_) sum_exp += std::exp(h.log_weight - max_lw);
        double log_norm = max_lw + std::log(sum_exp);
        for (auto& h : hypotheses_) h.log_weight -= log_norm;
    }

    void prune_hypotheses() {
        double log_thresh = std::log(prune_threshold_hyp_);
        std::vector<Hypothesis> kept;
        for (auto& h : hypotheses_) {
            if (h.log_weight >= log_thresh) kept.push_back(std::move(h));
        }
        if (kept.empty() && !hypotheses_.empty()) {
            auto best = std::max_element(hypotheses_.begin(), hypotheses_.end(),
                [](const auto& a, const auto& b) { return a.log_weight < b.log_weight; });
            kept.push_back(std::move(*best));
        }
        hypotheses_ = std::move(kept);
    }

    void cap_hypotheses() {
        if (static_cast<int>(hypotheses_.size()) <= max_hypotheses_) return;
        std::sort(hypotheses_.begin(), hypotheses_.end(),
            [](const auto& a, const auto& b) { return a.log_weight > b.log_weight; });
        hypotheses_.resize(static_cast<std::size_t>(max_hypotheses_));
    }

    void compact_track_table() {
        const std::size_t n = track_tab_.size();
        std::vector<bool> referenced(n, false);
        for (const auto& h : hypotheses_) {
            for (auto idx : h.track_indices) if (idx < n) referenced[idx] = true;
        }
        std::vector<std::size_t> new_idx(n, std::numeric_limits<std::size_t>::max());
        std::vector<std::unique_ptr<TrackEntry>> compact;
        std::size_t next = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (referenced[i]) {
                new_idx[i] = next++;
                compact.push_back(std::move(track_tab_[i]));
            }
        }
        for (auto& h : hypotheses_) {
            for (auto& idx : h.track_indices) idx = new_idx[idx];
        }
        track_tab_ = std::move(compact);
    }

    void prune_low_existence_tracks() {
        for (auto& h : hypotheses_) {
            std::vector<std::size_t> kept;
            kept.reserve(h.track_indices.size());
            for (auto idx : h.track_indices) {
                if (track_tab_[idx]->existence_probability >= prune_threshold_bern_) {
                    kept.push_back(idx);
                }
            }
            h.track_indices = std::move(kept);
        }
    }

    void compute_cardinality() {
        if (hypotheses_.empty()) {
            cardinality_pmf_ = Eigen::VectorXd::Zero(1);
            estimated_cardinality_ = 0.0;
            return;
        }
        int max_card = 0;
        auto count_existing = [&](const Hypothesis& h) {
            int c = 0;
            for (auto idx : h.track_indices) {
                if (track_tab_[idx]->existence_probability >= extract_threshold_) ++c;
            }
            return c;
        };
        for (const auto& h : hypotheses_) max_card = std::max(max_card, count_existing(h));
        cardinality_pmf_ = Eigen::VectorXd::Zero(max_card + 1);
        for (const auto& h : hypotheses_) {
            cardinality_pmf_(count_existing(h)) += std::exp(h.log_weight);
        }
        double s = cardinality_pmf_.sum();
        if (s > 0.0) cardinality_pmf_ /= s;
        estimated_cardinality_ = 0.0;
        for (int n = 0; n < cardinality_pmf_.size(); ++n) {
            estimated_cardinality_ += n * cardinality_pmf_(n);
        }
    }

    [[nodiscard]] std::size_t map_hypothesis_index() const {
        if (hypotheses_.empty()) return std::numeric_limits<std::size_t>::max();
        std::size_t idx = 0;
        for (std::size_t i = 1; i < hypotheses_.size(); ++i) {
            if (hypotheses_[i].log_weight > hypotheses_[idx].log_weight) idx = i;
        }
        return idx;
    }

    std::vector<std::unique_ptr<T>> extract_helper(const TrackEntry& trk) const {
        std::vector<std::unique_ptr<T>> states;
        states.reserve(trk.mixture_hist.size());
        for (const auto& mix : trk.mixture_hist) {
            if (mix->empty()) continue;
            std::size_t best = 0;
            double best_w = -1.0;
            for (std::size_t c = 0; c < mix->size(); ++c) {
                if (mix->weight(c) > best_w) { best_w = mix->weight(c); best = c; }
            }
            states.push_back(mix->component(best).clone_typed());
        }
        return states;
    }

    void update_extract_hist(std::size_t idx_cmp) {
        std::vector<std::vector<int>> used_meas(
            static_cast<std::size_t>(time_index_cntr_ + 1));
        std::vector<int> used_ids;
        std::vector<std::unique_ptr<ExtractedTrack>> new_hists;
        for (auto tab_idx : hypotheses_[idx_cmp].track_indices) {
            const auto& trk = *track_tab_[tab_idx];
            if (trk.existence_probability < extract_threshold_) continue;
            auto e = std::make_unique<ExtractedTrack>();
            e->id = trk.id;
            e->meas_assoc_hist = trk.meas_assoc_hist;
            e->b_time_index = trk.birth_time_index;
            e->states = extract_helper(trk);
            used_ids.push_back(trk.id);
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
        for (auto& existing : extracted_hists_) {
            bool claimed = false;
            for (int used : used_ids) {
                if (used == existing->id) { claimed = true; break; }
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
        extracted_hists_ = std::move(surviving);
    }

    void extract_states() {
        std::size_t idx = map_hypothesis_index();
        if (idx == std::numeric_limits<std::size_t>::max()) return;
        update_extract_hist(idx);
    }

    void push_extracted_snapshot() {
        if (this->max_history_ == 0) return;
        auto mix = std::make_unique<models::Mixture<T, MaxComponents>>();
        std::size_t idx = map_hypothesis_index();
        if (idx != std::numeric_limits<std::size_t>::max()) {
            for (auto tab_idx : hypotheses_[idx].track_indices) {
                if (tab_idx >= track_tab_.size()) continue;
                const auto& trk = *track_tab_[tab_idx];
                if (trk.existence_probability < extract_threshold_) continue;
                if (trk.mixture_hist.empty()) continue;
                const auto& last_mix = *trk.mixture_hist.back();
                if (last_mix.empty()) continue;
                std::size_t best = 0;
                double best_w = -1.0;
                for (std::size_t c = 0; c < last_mix.size(); ++c) {
                    if (last_mix.weight(c) > best_w) { best_w = last_mix.weight(c); best = c; }
                }
                mix->add_component(last_mix.component(best).clone_typed(),
                                   trk.existence_probability);
            }
        }
        this->push_history(extracted_mixtures_, std::move(mix));
    }

    MixturePtr predict_mixture(const models::Mixture<T, MaxComponents>& src, double dt) const {

        auto out = src.clone();
        filter_->predict_batch_dynamic(dt, *out);
        return out;
    }

    std::pair<MixturePtr, double> correct_mixture(
        const models::Mixture<T, MaxComponents>& src,
        const Eigen::VectorXd& meas_flat) const
    {
        auto out = std::make_unique<models::Mixture<T, MaxComponents>>();
        std::vector<double> w_new(src.size(), 0.0);
        double total_w = 0.0;
        for (std::size_t c = 0; c < src.size(); ++c) {
            auto [dist, qz] = filter_->correct(meas_flat, src.component(c));
            double w = src.weight(c) * qz;
            w_new[c] = w;
            total_w += w;
            out->add_component(dist.clone_typed(), w);
        }
        total_w += std::numeric_limits<double>::epsilon();
        for (std::size_t c = 0; c < out->size(); ++c) {
            out->weights()(static_cast<Eigen::Index>(c)) = w_new[c] / total_w;
        }
        return {std::move(out), total_w};
    }

    static Eigen::VectorXd flatten_measurement(const Eigen::MatrixXd& meas) {
        if (meas.cols() <= 1) return meas.col(0);
        Eigen::VectorXd out(meas.size());
        for (int c = 0; c < meas.cols(); ++c) {
            out.segment(c * meas.rows(), meas.rows()) = meas.col(c);
        }
        return out;
    }

    static Eigen::VectorXd gate_center(const Eigen::MatrixXd& meas, bool extended) {
        return (extended && meas.cols() > 1) ? Eigen::VectorXd(meas.rowwise().mean())
                                             : Eigen::VectorXd(meas.col(0));
    }

    bool passes_gate(const TrackEntry& trk, const Eigen::VectorXd& z_gate) const {
        const auto& mix = trk.current_mixture();
        for (std::size_t c = 0; c < mix.size(); ++c) {
            if (filter_->gate(z_gate, mix.component(c)) < gate_threshold_) return true;
        }
        return false;
    }

    std::vector<Eigen::MatrixXd> group_measurements(const Eigen::MatrixXd& measurements) const {
        std::vector<Eigen::MatrixXd> out;
        if (is_extended_ && cluster_obj_) {
            out = cluster_obj_->cluster(measurements);
        } else {
            for (int j = 0; j < measurements.cols(); ++j) out.push_back(measurements.col(j));
        }
        return out;
    }

    static std::vector<double> compute_kappa_vec(
        const std::vector<Eigen::MatrixXd>& meas_groups, double kappa_base)
    {
        std::vector<double> out(meas_groups.size());
        for (std::size_t j = 0; j < meas_groups.size(); ++j) {
            const int W = static_cast<int>(meas_groups[j].cols());
            out[j] = (W > 1) ? std::pow(kappa_base, W) : kappa_base;
        }
        return out;
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    bool has_filter_ = false;
    std::vector<std::unique_ptr<TrackEntry>> track_tab_;
    std::vector<Hypothesis> hypotheses_;
    std::vector<std::unique_ptr<ExtractedTrack>> extracted_hists_;
    std::deque<MixturePtr> extracted_mixtures_;

    double prune_threshold_hyp_ = 1e-3;
    double prune_threshold_bern_ = 1e-3;
    int max_hypotheses_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    bool gating_on_ = false;
    int k_best_ = 5;
    int next_track_id_ = 0;
    bool is_extended_ = false;
    std::shared_ptr<clustering::ClusterBase> cluster_obj_;

    Eigen::VectorXd cardinality_pmf_;
    double estimated_cardinality_ = 0.0;
    int time_index_cntr_ = 0;
};

}
