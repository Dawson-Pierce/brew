#pragma once

// Generic trajectory wrapper.
//
// Two orthogonal concerns are tracked side-by-side:
//
//   1. "Window" — a bounded ring buffer holding the most recent MaxWindow inner states
//      along with their stacked mean / stacked covariance. This is the working set
//      used by trajectory-aware filter math (prediction, correction, smoothing).
//      MaxWindow is a required compile-time constant; storage is fixed-capacity
//      (std::array<T, MaxWindow>, fixed Eigen of InnerDim*MaxWindow when InnerDim
//      is known at compile time, otherwise dynamic Eigen at runtime size state_dim*MaxWindow).
//
//   2. "State history" — an independent, unbounded-by-default record of every
//      finalized state this trajectory has held. Heap-backed std::vector<T>, runtime
//      cap via set_max_history() (0 disables, N keeps the most recent N, SIZE_MAX
//      keeps all). The window can be tiny (say 5 slots for smoother recursion) while
//      the history keeps everything for output or plotting — or vice versa.
//
// Ring-buffer semantics: when the window fills to MaxWindow, pushing a new step slides
// everything left by one inner-state block. The state history is *not* coupled to
// this sliding — it records the previous cycle's final state on each advance_window
// call regardless of window slides, subject only to its own independent cap.
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
// @mex_name TrajectoryIGGIW
// @mex_trajectory IGGIW
//
// @mex model
// @mex_name TrajectoryTemplatePose
// @mex_trajectory TemplatePose

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace brew::models {

/// Windowed trajectory of inner distribution T, plus an independent state history.
/// MaxWindow controls the ring-buffer window size (required compile-time constant);
/// state history is runtime-bounded.
template <typename T, int MaxWindow>
class Trajectory {
public:
    using Scalar = typename T::Vector::Scalar;
    using Vector = typename T::Vector;
    using Matrix = typename T::Matrix;
    static constexpr int InnerDim = Vector::RowsAtCompileTime;

    static constexpr int StackedDim =
        (InnerDim == Eigen::Dynamic)
            ? Eigen::Dynamic
            : InnerDim * MaxWindow;

    using StackedVector = Eigen::Matrix<Scalar, StackedDim, 1>;
    using StackedMatrix = Eigen::Matrix<Scalar, StackedDim, StackedDim>;

    using WindowStorage = std::array<T, (MaxWindow > 0 ? MaxWindow : 1)>;

    int state_dim = 0;

    Trajectory() {
        stacked_mean_.setZero();
        stacked_covariance_.setZero();
    }

    Trajectory(int state_dim_, T initial) : state_dim(state_dim_) {
        if constexpr (StackedDim == Eigen::Dynamic) {
            stacked_mean_ = StackedVector::Zero(state_dim_ * MaxWindow);
            stacked_covariance_ = StackedMatrix::Zero(state_dim_ * MaxWindow, state_dim_ * MaxWindow);
        } else {
            stacked_mean_.setZero();
            stacked_covariance_.setZero();
        }
        stacked_mean_.segment(0, state_dim_) = initial.mean();
        stacked_covariance_.block(0, 0, state_dim_, state_dim_) = initial.covariance();
        history_[0] = initial;
        window_size_ = 1;
        // Seed the state history with the initial state, respecting max_history_.
        push_state_history(std::move(initial));
    }

    [[nodiscard]] std::unique_ptr<Trajectory> clone() const {
        return std::make_unique<Trajectory>(*this);
    }

    [[nodiscard]] std::unique_ptr<Trajectory> clone_typed() const { return clone(); }

    // Window metadata

    /// Live step count in the window (advanced by advance_window).
    [[nodiscard]] int window_size() const { return window_size_; }
    [[nodiscard]] static constexpr int max_window_size() { return MaxWindow; }
    [[nodiscard]] int stacked_size() const { return state_dim * window_size_; }
    [[nodiscard]] bool is_full() const { return window_size_ == MaxWindow; }

    [[nodiscard]] bool is_extended() const {
        if (window_size_ == 0) return false;
        return current().is_extended();
    }

    // Stacked state access
    // The stored stacked_mean_/covariance_ are always at full max size,
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

    // Window state access (the bounded ring-buffer of T values inside the window).
    // These are the filter's working set; for the full-lifetime record use state_history().

    [[nodiscard]] const WindowStorage& history() const { return history_; }
    [[nodiscard]] WindowStorage& history() { return history_; }

    [[nodiscard]] const T& current() const { return history_[window_size_ - 1]; }
    [[nodiscard]] T& current() { return history_[window_size_ - 1]; }

    [[nodiscard]] const T& history_at(int i) const { return history_[i]; }
    [[nodiscard]] T& history_at(int i) { return history_[i]; }

    // State-history access (independent, full-lifetime record).

    /// Cap the state history.
    /// Semantics:
    ///   0           - disable recording (existing entries are cleared)
    ///   N > 0       - keep only the N most recent states (trims immediately)
    ///   SIZE_MAX    - unbounded (default)
    void set_max_history(std::size_t n) {
        max_history_ = n;
        if (n == 0) {
            state_history_.clear();
            return;
        }
        while (state_history_.size() > n) state_history_.erase(state_history_.begin());
    }

    [[nodiscard]] std::size_t max_history() const { return max_history_; }
    [[nodiscard]] const std::vector<T>& state_history() const { return state_history_; }
    [[nodiscard]] std::vector<T>& state_history() { return state_history_; }

    // Ring-buffer advance

    /// Make room for a new step at the end of the window. After this returns,
    /// `last_index()` points at a zeroed slot ready to be filled with a prediction.
    ///
    /// Window behaviour:
    /// - not full: increments window_size_; last slot zeroed.
    /// - full: slides stacked_mean_/covariance_/history_ left by one inner-state block,
    ///   last slot zeroed, window_size_ unchanged (== MaxWindow).
    /// Returns true if a slide occurred (oldest window slot was dropped).
    ///
    /// State history behaviour: on every call except the very first post-construction,
    /// pushes the current (last-finalized) window state to state_history_, trimming
    /// front entries to respect max_history_. The first call is a no-op for history
    /// because the constructor already seeded it with the initial state.
    bool advance_window() {
        if (advance_count_ > 0 && window_size_ > 0) {
            push_state_history(current());
        }
        ++advance_count_;

        const int sd = state_dim;

        // Grow / bump counter (below cap)
        if (window_size_ < MaxWindow) {
            ++window_size_;
            mean_at(window_size_ - 1).setZero();
            cov_at(window_size_ - 1, window_size_ - 1).setZero();
            const int prev = (window_size_ - 1) * sd;
            stacked_covariance_.block(prev, 0, sd, prev).setZero();
            stacked_covariance_.block(0, prev, prev, sd).setZero();
            history_[window_size_ - 1] = T{};
            return false;
        }

        // At cap: slide window left by one block (drop oldest), zero new tail slot
        const int kept = (MaxWindow - 1) * sd;
        stacked_mean_.head(kept) = stacked_mean_.tail(kept).eval();
        stacked_mean_.tail(sd).setZero();
        stacked_covariance_.topLeftCorner(kept, kept) =
            stacked_covariance_.bottomRightCorner(kept, kept).eval();
        stacked_covariance_.rightCols(sd).setZero();
        stacked_covariance_.bottomRows(sd).setZero();
        for (int i = 0; i < MaxWindow - 1; ++i) {
            history_[i] = std::move(history_[i + 1]);
        }
        history_[MaxWindow - 1] = T{};
        return true;
    }

    /// Explicit commit: push current() to state_history_. Useful after the final
    /// correct() when no further advance_window will occur to flush the last state.
    void commit_current_to_state_history() {
        if (window_size_ > 0) push_state_history(current());
    }

    [[nodiscard]] int last_index() const { return window_size_ - 1; }

    // Convenience: extract last-step views

    [[nodiscard]] Vector get_last_state() const {
        return mean_at(window_size_ - 1);
    }
    [[nodiscard]] Matrix get_last_cov() const {
        const int li = window_size_ - 1;
        return cov_at(li, li);
    }

    /// Returns (state_dim x window_size) matrix of window-slot means (from history_).
    [[nodiscard]] Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> mean_history() const {
        const int ws = window_size_;
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
        const int ws = window_size_;
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
    void push_state_history(T v) {
        if (max_history_ == 0) return;
        state_history_.push_back(std::move(v));
        while (state_history_.size() > max_history_) {
            state_history_.erase(state_history_.begin());
        }
    }

    StackedVector stacked_mean_{};
    StackedMatrix stacked_covariance_{};
    WindowStorage history_{};
    int window_size_ = 0;

    // Independent full-lifetime record.
    std::vector<T> state_history_{};
    std::size_t max_history_ = std::numeric_limits<std::size_t>::max();
    int advance_count_ = 0;
};

// Type trait for detecting Trajectory<T, M>
template <typename U>
struct is_trajectory : std::false_type {};

template <typename U, int M>
struct is_trajectory<Trajectory<U, M>> : std::true_type {};

} // namespace brew::models
