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
//
//
//
//

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace brew::models {

template <typename T>
class TrajectoryWindow {
public:
    using InnerType = T;
    using Scalar = typename T::Vector::Scalar;
    using Vector = typename T::Vector;
    using Matrix = typename T::Matrix;
    static constexpr int InnerDim = Vector::RowsAtCompileTime;

    using StackedVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using StackedMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using WindowStorage = std::vector<T>;

    static constexpr int kDefaultWindow = 5;

    int state_dim = 0;

    TrajectoryWindow() {
        history_.assign(max_window_, T{});
    }

    TrajectoryWindow(int state_dim_, T initial, int max_window = kDefaultWindow)
        : state_dim(state_dim_), max_window_(max_window > 0 ? max_window : 1) {
        stacked_mean_ = StackedVector::Zero(state_dim_ * max_window_);
        stacked_covariance_ = StackedMatrix::Zero(state_dim_ * max_window_, state_dim_ * max_window_);
        history_.assign(max_window_, T{});
        stacked_mean_.segment(0, state_dim_) = initial.mean();
        stacked_covariance_.block(0, 0, state_dim_, state_dim_) = initial.covariance();
        history_[0] = initial;
        window_size_ = 1;

        push_state_history(std::move(initial));
    }

    [[nodiscard]] std::unique_ptr<TrajectoryWindow> clone() const {
        return std::make_unique<TrajectoryWindow>(*this);
    }

    [[nodiscard]] std::unique_ptr<TrajectoryWindow> clone_typed() const { return clone(); }

    [[nodiscard]] int window_size() const { return window_size_; }
    [[nodiscard]] int max_window_size() const { return max_window_; }
    [[nodiscard]] int stacked_size() const { return state_dim * window_size_; }
    [[nodiscard]] bool is_full() const { return window_size_ == max_window_; }

    void set_max_window_size(int n) {
        max_window_ = (n > 0 ? n : 1);
        history_.assign(max_window_, T{});
        if (state_dim > 0) {
            stacked_mean_ = StackedVector::Zero(state_dim * max_window_);
            stacked_covariance_ = StackedMatrix::Zero(state_dim * max_window_, state_dim * max_window_);
        }
        window_size_ = 0;
    }
    void set_window_size(int n) { set_max_window_size(n); }

    [[nodiscard]] bool is_extended() const {
        if (window_size_ == 0) return false;
        return current().is_extended();
    }

    [[nodiscard]] const StackedVector& mean() const { return stacked_mean_; }
    [[nodiscard]] StackedVector& mean() { return stacked_mean_; }
    [[nodiscard]] const StackedMatrix& covariance() const { return stacked_covariance_; }
    [[nodiscard]] StackedMatrix& covariance() { return stacked_covariance_; }

    [[nodiscard]] auto mean_at(int i) { return stacked_mean_.segment(i * state_dim, state_dim); }
    [[nodiscard]] auto mean_at(int i) const { return stacked_mean_.segment(i * state_dim, state_dim); }

    [[nodiscard]] auto cov_at(int i, int j) {
        return stacked_covariance_.block(i * state_dim, j * state_dim, state_dim, state_dim);
    }
    [[nodiscard]] auto cov_at(int i, int j) const {
        return stacked_covariance_.block(i * state_dim, j * state_dim, state_dim, state_dim);
    }

    [[nodiscard]] const WindowStorage& history() const { return history_; }
    [[nodiscard]] WindowStorage& history() { return history_; }

    [[nodiscard]] const T& current() const { return history_[window_size_ - 1]; }
    [[nodiscard]] T& current() { return history_[window_size_ - 1]; }

    [[nodiscard]] const T& history_at(int i) const { return history_[i]; }
    [[nodiscard]] T& history_at(int i) { return history_[i]; }

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

    bool advance_window() {
        if (advance_count_ > 0 && window_size_ > 0) {
            push_state_history(current());
        }
        ++advance_count_;

        const int sd = state_dim;

        if (window_size_ < max_window_) {
            ++window_size_;
            mean_at(window_size_ - 1).setZero();
            cov_at(window_size_ - 1, window_size_ - 1).setZero();
            const int prev = (window_size_ - 1) * sd;
            stacked_covariance_.block(prev, 0, sd, prev).setZero();
            stacked_covariance_.block(0, prev, prev, sd).setZero();
            history_[window_size_ - 1] = T{};
            return false;
        }

        const int kept = (max_window_ - 1) * sd;
        stacked_mean_.head(kept) = stacked_mean_.tail(kept).eval();
        stacked_mean_.tail(sd).setZero();
        stacked_covariance_.topLeftCorner(kept, kept) =
            stacked_covariance_.bottomRightCorner(kept, kept).eval();
        stacked_covariance_.rightCols(sd).setZero();
        stacked_covariance_.bottomRows(sd).setZero();
        for (int i = 0; i < max_window_ - 1; ++i) {
            history_[i] = std::move(history_[i + 1]);
        }
        history_[max_window_ - 1] = T{};
        return true;
    }

    void commit_current_to_state_history() {
        if (window_size_ > 0) push_state_history(current());
    }

    [[nodiscard]] int last_index() const { return window_size_ - 1; }

    [[nodiscard]] Vector get_last_state() const {
        return mean_at(window_size_ - 1);
    }
    [[nodiscard]] Matrix get_last_cov() const {
        const int li = window_size_ - 1;
        return cov_at(li, li);
    }

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

    [[nodiscard]] Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> state_history_means() const {
        const int m = static_cast<int>(state_history_.size());
        if (m == 0 || state_dim <= 0) {
            return Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic>();
        }
        Eigen::Matrix<Scalar, InnerDim, Eigen::Dynamic> result(state_dim, m);
        for (int i = 0; i < m; ++i) {
            result.col(i) = state_history_[i].mean();
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
    int max_window_ = kDefaultWindow;

    std::vector<T> state_history_{};
    std::size_t max_history_ = std::numeric_limits<std::size_t>::max();
    int advance_count_ = 0;
};

template <typename U>
struct is_trajectory : std::false_type {};

template <typename U>
struct is_trajectory<TrajectoryWindow<U>> : std::true_type {};

}
