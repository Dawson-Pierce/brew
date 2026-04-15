#pragma once

// Generic trajectory wrapper — windowed stacked state + full history of inner models.
// One class handles Gaussian / GGIW / GGIWOrientation / TemplatePose trajectories,
// following the same pattern as Mixture<T>.
//
// Ring-buffer semantics: when window fills to MaxHistory, pushing a new step slides
// everything left by one inner-state block. When MaxHistory is Eigen::Dynamic the
// stacked state grows; the filter's l_window_ provides a runtime soft cap.
//
// @mex model
// @mex_name TrajectoryGaussian
// @mex_trajectory Gaussian
//
// @mex model
// @mex_name TrajectoryGGIW
// @mex_trajectory GGIW
//
// @mex model
// @mex_name TrajectoryGGIWOrientation
// @mex_trajectory GGIWOrientation
//
// @mex model
// @mex_name TrajectoryTemplatePose
// @mex_trajectory TemplatePose

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace brew::models {

/// Windowed trajectory of inner distribution T.
/// - MaxHistory = Eigen::Dynamic: heap-backed (std::vector<T> + dynamic Eigen); unlimited window.
/// - MaxHistory fixed: stack-backed (std::array<T, MaxHistory> + fixed Eigen of D*MaxHistory).
template <typename T, int MaxHistory = Eigen::Dynamic>
class Trajectory {
public:
    using Scalar = typename T::Vector::Scalar;
    using Vector = typename T::Vector;
    using Matrix = typename T::Matrix;
    static constexpr int InnerDim = Vector::RowsAtCompileTime;

    static constexpr bool fixed_history = (MaxHistory != Eigen::Dynamic);

    static constexpr int StackedDim =
        (InnerDim == Eigen::Dynamic || !fixed_history)
            ? Eigen::Dynamic
            : InnerDim * MaxHistory;

    using StackedVector = Eigen::Matrix<Scalar, StackedDim, 1>;
    using StackedMatrix = Eigen::Matrix<Scalar, StackedDim, StackedDim>;

    using HistoryStorage = std::conditional_t<fixed_history,
        std::array<T, (MaxHistory > 0 ? MaxHistory : 1)>,
        std::vector<T>>;

    int state_dim = 0;

    Trajectory() {
        if constexpr (fixed_history) {
            stacked_mean_.setZero();
            stacked_covariance_.setZero();
        }
    }

    Trajectory(int state_dim_, T initial) : state_dim(state_dim_) {
        if constexpr (fixed_history) {
            stacked_mean_.setZero();
            stacked_covariance_.setZero();
            stacked_mean_.segment(0, state_dim_) = initial.mean();
            stacked_covariance_.block(0, 0, state_dim_, state_dim_) = initial.covariance();
            history_[0] = std::move(initial);
            window_size_ = 1;
        } else {
            stacked_mean_ = initial.mean();
            stacked_covariance_ = initial.covariance();
            history_.push_back(std::move(initial));
            window_size_ = 1;
        }
    }

    [[nodiscard]] std::unique_ptr<Trajectory> clone() const {
        return std::make_unique<Trajectory>(*this);
    }

    [[nodiscard]] std::unique_ptr<Trajectory> clone_typed() const { return clone(); }

    // Window metadata

    /// Live step count. For fixed_history, uses the internal window_size_ counter
    /// (advanced by advance_window). For dynamic, derives from history_.size() so
    /// callers that populate history_ directly (e.g. via history().push_back(...)
    /// for bespoke construction) see a consistent count.
    [[nodiscard]] int window_size() const {
        if constexpr (fixed_history) return window_size_;
        else return static_cast<int>(history_.size());
    }
    [[nodiscard]] static constexpr int max_window_size() {
        return fixed_history ? MaxHistory : -1;
    }
    [[nodiscard]] int stacked_size() const { return state_dim * window_size(); }
    [[nodiscard]] bool is_full() const {
        return fixed_history && window_size_ == MaxHistory;
    }

    [[nodiscard]] bool is_extended() const {
        if (window_size() == 0) return false;
        return current().is_extended();
    }

    // Stacked state access
    // The stored stacked_mean_/covariance_ are always at full max size when fixed_history,
    // zero-padded beyond the live window.

    [[nodiscard]] const StackedVector& mean() const { return stacked_mean_; }
    [[nodiscard]] StackedVector& mean() { return stacked_mean_; }
    [[nodiscard]] const StackedMatrix& covariance() const { return stacked_covariance_; }
    [[nodiscard]] StackedMatrix& covariance() { return stacked_covariance_; }

    // Per-step block views

    [[nodiscard]] auto mean_at(int i) { return stacked_mean_.segment(i * state_dim, state_dim); }
    [[nodiscard]] auto mean_at(int i) const { return stacked_mean_.segment(i * state_dim, state_dim); }

    [[nodiscard]] auto cov_at(int i, int j) {
        return stacked_covariance_.block(i * state_dim, j * state_dim, state_dim, state_dim);
    }
    [[nodiscard]] auto cov_at(int i, int j) const {
        return stacked_covariance_.block(i * state_dim, j * state_dim, state_dim, state_dim);
    }

    // History access

    [[nodiscard]] const HistoryStorage& history() const { return history_; }
    [[nodiscard]] HistoryStorage& history() { return history_; }

    [[nodiscard]] const T& current() const {
        if constexpr (fixed_history) return history_[window_size_ - 1];
        else return history_.back();  // reads from vector, ignores counter
    }
    [[nodiscard]] T& current() {
        if constexpr (fixed_history) return history_[window_size_ - 1];
        else return history_.back();
    }

    [[nodiscard]] const T& history_at(int i) const { return history_[i]; }
    [[nodiscard]] T& history_at(int i) { return history_[i]; }

    // Ring-buffer advance

    /// Make room for a new step at the end. After this returns, `last_index()` points
    /// at a zeroed slot ready to be filled with a prediction.
    ///
    /// Behaviour:
    /// - fixed_history and not full: increments window_size_; last slot zeroed.
    /// - fixed_history and full: slides stacked_mean_/covariance_/history_ left by one
    ///   inner-state block, last slot zeroed, window_size_ unchanged (== MaxHistory).
    /// - dynamic: if `soft_cap > 0` and window_size_ >= soft_cap, slides left instead of
    ///   growing; otherwise grows the stacked matrices by one inner block.
    /// Returns true if a slide occurred (oldest was dropped).
    bool advance_window(int soft_cap = -1) {
        const int sd = state_dim;
        const int cap = fixed_history ? MaxHistory
                       : (soft_cap > 0 ? soft_cap : std::numeric_limits<int>::max());
        const int ws = window_size();

        // Grow / bump counter (below cap)
        if (ws < cap) {
            if constexpr (fixed_history) {
                // Fixed storage: bump the live-slot counter; zero the new slot
                // (might contain stale data from a prior slide).
                ++window_size_;
                mean_at(window_size_ - 1).setZero();
                cov_at(window_size_ - 1, window_size_ - 1).setZero();
                const int prev = (window_size_ - 1) * sd;
                stacked_covariance_.block(prev, 0, sd, prev).setZero();
                stacked_covariance_.block(0, prev, prev, sd).setZero();
                history_[window_size_ - 1] = T{};
            } else {
                // Dynamic storage: reallocate stacked_mean_/covariance_ to new size.
                // Source size from stacked_mean_ directly so bespoke manual init
                // (set mean()/covariance()/history directly without advance_window)
                // still works.
                const int old_total = static_cast<int>(stacked_mean_.size());
                const int new_total = old_total + sd;
                StackedVector new_mean = StackedVector::Zero(new_total);
                new_mean.head(old_total) = stacked_mean_;
                stacked_mean_ = std::move(new_mean);

                StackedMatrix new_cov = StackedMatrix::Zero(new_total, new_total);
                new_cov.topLeftCorner(old_total, old_total) = stacked_covariance_;
                stacked_covariance_ = std::move(new_cov);

                history_.emplace_back();
                window_size_ = static_cast<int>(history_.size());
            }
            return false;
        }

        // At cap: slide left by one block (drop oldest), zero new tail slot
        const int kept = (cap - 1) * sd;
        stacked_mean_.head(kept) = stacked_mean_.tail(kept).eval();
        stacked_mean_.tail(sd).setZero();
        stacked_covariance_.topLeftCorner(kept, kept) =
            stacked_covariance_.bottomRightCorner(kept, kept).eval();
        stacked_covariance_.rightCols(sd).setZero();
        stacked_covariance_.bottomRows(sd).setZero();
        if constexpr (fixed_history) {
            for (int i = 0; i < MaxHistory - 1; ++i) {
                history_[i] = std::move(history_[i + 1]);
            }
            history_[MaxHistory - 1] = T{};
        } else {
            history_.erase(history_.begin());
            history_.emplace_back();
        }
        return true;
    }

    [[nodiscard]] int last_index() const { return window_size() - 1; }

    // Convenience: extract last-step views

    [[nodiscard]] Vector get_last_state() const {
        return mean_at(window_size() - 1);
    }
    [[nodiscard]] Matrix get_last_cov() const {
        const int li = window_size() - 1;
        return cov_at(li, li);
    }

    /// Returns (state_dim x window_size) matrix of history means.
    [[nodiscard]] Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> mean_history() const {
        const int ws = window_size();
        if (ws == 0 || state_dim <= 0) {
            return Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic>();
        }
        Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> result(state_dim, ws);
        for (int i = 0; i < ws; ++i) {
            result.col(i) = history_[i].mean();
        }
        return result;
    }

    /// Returns (state_dim x window_size) matrix of stacked-state per-step blocks.
    [[nodiscard]] Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> rearrange_states() const {
        const int ws = window_size();
        if (state_dim <= 0 || ws == 0) {
            return Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic>();
        }
        Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> result(state_dim, ws);
        for (int i = 0; i < ws; ++i) {
            result.col(i) = mean_at(i);
        }
        return result;
    }

private:
    StackedVector stacked_mean_{};
    StackedMatrix stacked_covariance_{};
    HistoryStorage history_{};
    int window_size_ = 0;
};

// Type trait for detecting Trajectory<T, ...>
template <typename U>
struct is_trajectory : std::false_type {};

template <typename U, int M>
struct is_trajectory<Trajectory<U, M>> : std::true_type {};

} // namespace brew::models
