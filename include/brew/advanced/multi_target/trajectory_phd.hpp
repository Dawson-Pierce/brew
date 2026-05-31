#pragma once

#include "brew/advanced/multi_target/phd.hpp"
#include "brew/core/models/trajectory.hpp"

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <vector>

namespace brew::multi_target {

/// PHD variant for trajectory distributions that retains the trajectories of
/// tracks that have died. Live tracking is identical to PHD; on each cleanup,
/// just before pruning, any component that is about to be pruned (weight below
/// the prune threshold) whose recorded state history is at least
/// death_min_length long is copied into a persistent dead-track store.
///
/// track_histories() returns the state-history path of every track -- the
/// retained dead tracks first, then the currently-extracted live tracks -- so
/// the full set of trajectories (including ones that have ended) is available
/// for plotting/analysis.
///
/// Non-trajectory T compiles and behaves exactly like a plain PHD (the retain
/// logic is guarded by is_trajectory, so nothing extra is recorded).
// @mex rfs
// @mex_name TrajectoryPHD
// @mex_params prune_threshold:double:1e-4, merge_threshold:double:4.0, max_components:int:100, extract_threshold:double:0.5, gate_threshold:double:9.0, death_min_length:int:2
// @mex_init set_intensity
// @mex_has birth_weights, cluster_object, track_histories
template <typename T, int MaxComponents = Eigen::Dynamic>
class TrajectoryPHD : public PHD<T, MaxComponents> {
public:
    using Base = PHD<T, MaxComponents>;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<TrajectoryPHD<T, MaxComponents>>();
        c->prob_detection_ = this->prob_detection_;
        c->prob_survive_ = this->prob_survive_;
        c->clutter_rate_ = this->clutter_rate_;
        c->clutter_density_ = this->clutter_density_;
        if (this->filter_) c->filter_ = this->filter_->clone();
        if (this->intensity_) c->intensity_ = this->intensity_->clone();
        if (this->birth_model_) c->birth_model_ = this->birth_model_->clone();
        c->prune_threshold_ = this->prune_threshold_;
        c->merge_threshold_ = this->merge_threshold_;
        c->max_components_ = this->max_components_;
        c->extract_threshold_ = this->extract_threshold_;
        c->gate_threshold_ = this->gate_threshold_;
        c->is_extended_ = this->is_extended_;
        if (this->cluster_obj_) c->cluster_obj_ = this->cluster_obj_;
        c->death_min_length_ = death_min_length_;
        for (const auto& d : dead_) c->dead_.push_back(d->clone_typed());
        return c;
    }

    void set_death_min_length(int n) { death_min_length_ = n; }
    [[nodiscard]] int death_min_length() const { return death_min_length_; }

    /// Number of retained dead tracks.
    [[nodiscard]] std::size_t dead_count() const { return dead_.size(); }

    void cleanup() override {
        if constexpr (models::is_trajectory<T>::value) {
            capture_dead();
        }
        Base::cleanup();
    }

    /// State-history paths of all tracks: retained dead tracks first, then the
    /// currently-extracted live tracks. Each entry maps a synthetic id to the
    /// sequence of per-step state means.
    [[nodiscard]] std::map<int, std::vector<Eigen::VectorXd>> track_histories() const {
        std::map<int, std::vector<Eigen::VectorXd>> out;
        if constexpr (models::is_trajectory<T>::value) {
            int id = 0;
            for (const auto& d : dead_) {
                out[id++] = history_states(*d);
            }
            if (this->intensity_) {
                for (std::size_t i = 0; i < this->intensity_->size(); ++i) {
                    if (this->intensity_->weight(i) >= this->extract_threshold_) {
                        out[id++] = history_states(this->intensity_->component(i));
                    }
                }
            }
        }
        return out;
    }

private:
    void capture_dead() {
        if (!this->intensity_) return;
        for (std::size_t i = 0; i < this->intensity_->size(); ++i) {
            if (this->intensity_->weight(i) < this->prune_threshold_) {
                const auto& comp = this->intensity_->component(i);
                if (static_cast<int>(comp.state_history().size()) >= death_min_length_) {
                    dead_.push_back(comp.clone_typed());
                }
            }
        }
    }

    static std::vector<Eigen::VectorXd> history_states(const T& traj) {
        const Eigen::MatrixXd sh = traj.state_history_means();
        std::vector<Eigen::VectorXd> states;
        states.reserve(static_cast<std::size_t>(sh.cols()));
        for (Eigen::Index j = 0; j < sh.cols(); ++j) {
            states.push_back(sh.col(j));
        }
        return states;
    }

    std::vector<std::unique_ptr<T>> dead_;
    int death_min_length_ = 2;
};

} // namespace brew::multi_target
