#pragma once

// Ported from: +BREW/+distributions/TrajectoryGaussian.m
// Original name: TrajectoryGaussian
// Ported on: 2026-02-07
// Notes: Pure data — no sampling, no plotting.

#include "brew/models/trajectory_base_model.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace brew::models {

/// Gaussian trajectory distribution.
/// Stores a windowed trajectory of states as a stacked mean/covariance.
class TrajectoryGaussian : public TrajectoryBaseModel {
public:
    TrajectoryGaussian() = default;

    inline TrajectoryGaussian(int idx, int state_dim,
                              Eigen::VectorXd mean, Eigen::MatrixXd covariance)
        : TrajectoryBaseModel(idx, state_dim),
          mean_(std::move(mean)),
          covariance_(std::move(covariance)) {
        const int steps = (state_dim > 0) ? static_cast<int>(mean_.size() / state_dim) : 0;
        window_size = std::max(1, steps);
        if (state_dim > 0 && mean_.size() > 0 && steps > 0) {
            mean_history_ = Eigen::Map<const Eigen::MatrixXd>(
                mean_.data(), state_dim, steps);
        } else {
            mean_history_.resize(0, 0);
        }
        cov_history_.clear();
        cov_history_.push_back(covariance_);
    }

    [[nodiscard]] inline std::unique_ptr<TrajectoryGaussian> clone() const {
        auto c = std::make_unique<TrajectoryGaussian>(init_idx, state_dim, mean_, covariance_);
        c->window_size = window_size;
        c->mean_history_ = mean_history_;
        c->cov_history_ = cov_history_;
        return c;
    }

    [[nodiscard]] std::unique_ptr<TrajectoryGaussian> clone_typed() const { return clone(); }

    // ---- Data access ----
    [[nodiscard]] const Eigen::VectorXd& mean() const { return mean_; }
    [[nodiscard]] Eigen::VectorXd& mean() { return mean_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return covariance_; }
    [[nodiscard]] Eigen::MatrixXd& covariance() { return covariance_; }
    [[nodiscard]] const Eigen::MatrixXd& mean_history() const { return mean_history_; }
    [[nodiscard]] Eigen::MatrixXd& mean_history() { return mean_history_; }
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& cov_history() const { return cov_history_; }
    [[nodiscard]] std::vector<Eigen::MatrixXd>& cov_history() { return cov_history_; }

    [[nodiscard]] inline Eigen::VectorXd get_last_state_mean() const {
        return mean_.tail(state_dim);
    }

    [[nodiscard]] inline Eigen::MatrixXd get_last_state_covariance() const {
        const int n = state_dim;
        const int total = static_cast<int>(mean_.size());
        return covariance_.block(total - n, total - n, n, n);
    }

    [[nodiscard]] inline Eigen::VectorXd get_last_state() const {
        return get_last_state_mean();
    }

    [[nodiscard]] inline Eigen::MatrixXd get_last_cov() const {
        return get_last_state_covariance();
    }

    [[nodiscard]] inline Eigen::MatrixXd rearrange_states() const {
        if (state_dim <= 0) return Eigen::MatrixXd();
        const int steps = static_cast<int>(mean_.size() / state_dim);
        return Eigen::Map<const Eigen::MatrixXd>(mean_.data(), state_dim, steps);
    }

private:
    Eigen::VectorXd mean_;
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd mean_history_;
    std::vector<Eigen::MatrixXd> cov_history_;
};

} // namespace brew::models
