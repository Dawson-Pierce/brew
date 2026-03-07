#pragma once

// Generic trajectory wrapper — windowed stacked state + full history of inner models.
// Replaces TrajectoryBaseModel, TrajectoryGaussian, TrajectoryGGIW, TrajectoryGGIWOrientation
// with a single template, following the same pattern as Mixture<T>.

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace brew::models {

template <typename T>
class Trajectory {
public:
    int state_dim = 0;

    Trajectory() = default;

    Trajectory(int state_dim, T initial)
        : state_dim(state_dim),
          stacked_mean_(initial.mean()),
          stacked_covariance_(initial.covariance()) {
        history_.push_back(std::move(initial));
    }

    [[nodiscard]] std::unique_ptr<Trajectory<T>> clone() const {
        auto c = std::make_unique<Trajectory<T>>();
        c->state_dim = state_dim;
        c->stacked_mean_ = stacked_mean_;
        c->stacked_covariance_ = stacked_covariance_;
        c->history_ = history_;
        return c;
    }

    [[nodiscard]] std::unique_ptr<Trajectory<T>> clone_typed() const { return clone(); }

    // ---- Trajectory metadata ----

    [[nodiscard]] int window_size() const { return static_cast<int>(history_.size()); }

    [[nodiscard]] int stacked_size() const {
        return (state_dim > 0) ? static_cast<int>(stacked_mean_.size()) / state_dim : 0;
    }

    [[nodiscard]] bool is_extended() const {
        if (history_.empty()) return false;
        return history_.back().is_extended();
    }

    // ---- Stacked state access (for filter math) ----

    [[nodiscard]] const Eigen::VectorXd& mean() const { return stacked_mean_; }
    [[nodiscard]] Eigen::VectorXd& mean() { return stacked_mean_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return stacked_covariance_; }
    [[nodiscard]] Eigen::MatrixXd& covariance() { return stacked_covariance_; }

    // ---- History access ----

    [[nodiscard]] const std::vector<T>& history() const { return history_; }
    [[nodiscard]] std::vector<T>& history() { return history_; }

    [[nodiscard]] const T& current() const { return history_.back(); }
    [[nodiscard]] T& current() { return history_.back(); }

    // ---- Computed from history (on the fly) ----

    [[nodiscard]] Eigen::MatrixXd mean_history() const {
        if (history_.empty() || state_dim <= 0) return Eigen::MatrixXd();
        Eigen::MatrixXd result(state_dim, static_cast<int>(history_.size()));
        for (int i = 0; i < static_cast<int>(history_.size()); ++i) {
            result.col(i) = history_[i].mean();
        }
        return result;
    }

    // ---- Convenience: extract from stacked state ----

    [[nodiscard]] Eigen::VectorXd get_last_state() const {
        return stacked_mean_.tail(state_dim);
    }

    [[nodiscard]] Eigen::MatrixXd get_last_cov() const {
        const int total = static_cast<int>(stacked_mean_.size());
        return stacked_covariance_.block(total - state_dim, total - state_dim,
                                         state_dim, state_dim);
    }

    [[nodiscard]] Eigen::MatrixXd rearrange_states() const {
        if (state_dim <= 0) return Eigen::MatrixXd();
        const int steps = static_cast<int>(stacked_mean_.size() / state_dim);
        return Eigen::Map<const Eigen::MatrixXd>(stacked_mean_.data(), state_dim, steps);
    }

private:
    Eigen::VectorXd stacked_mean_;
    Eigen::MatrixXd stacked_covariance_;
    std::vector<T> history_;
};

// Type trait for detecting Trajectory<T>
template <typename U>
struct is_trajectory : std::false_type {};

template <typename U>
struct is_trajectory<Trajectory<U>> : std::true_type {};

} // namespace brew::models
