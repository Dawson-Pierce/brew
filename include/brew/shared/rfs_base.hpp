#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <memory>

namespace brew::multi_target {

/// Abstract base class for Random Finite Set filters.

class RFSBase {
public:
    virtual ~RFSBase() = default;

    [[nodiscard]] virtual std::unique_ptr<RFSBase> clone() const = 0;

    // ---- Configuration ----

    void set_prob_detection(double pd) { prob_detection_ = pd; }
    void set_prob_survive(double ps) { prob_survive_ = ps; }
    void set_clutter_rate(double rate) { clutter_rate_ = rate; }
    void set_clutter_density(double density) { clutter_density_ = density; }

    /// Cap on per-step history containers (e.g. extracted_mixtures, cardinality_history).
    /// Semantics:
    ///   0           - disable recording entirely (hardware-friendly: zero memory growth)
    ///   N > 0       - ring-buffer keeping only the N most recent entries
    ///   SIZE_MAX    - unbounded (default; keeps everything, grows forever)
    /// Trimming happens on the next cleanup(); lowering the cap does not retroactively
    /// trim existing history until another snapshot is produced.
    void set_max_history(std::size_t n) { max_history_ = n; }

    [[nodiscard]] double prob_detection() const { return prob_detection_; }
    [[nodiscard]] double prob_survive() const { return prob_survive_; }
    [[nodiscard]] double clutter_rate() const { return clutter_rate_; }
    [[nodiscard]] double clutter_density() const { return clutter_density_; }
    [[nodiscard]] std::size_t max_history() const { return max_history_; }

    // ---- Abstract interface ----

    virtual void predict(int timestep, double dt) = 0;
    virtual void correct(const Eigen::MatrixXd& measurements) = 0;
    virtual void cleanup() = 0;

protected:
    /// Shared helper: push onto a history container, enforce max_history_ cap.
    /// Callers pass a deque (front-pop is O(1)); for cap==0 nothing is stored.
    template <typename Container, typename Value>
    void push_history(Container& c, Value&& v) const {
        if (max_history_ == 0) return;
        c.push_back(std::forward<Value>(v));
        while (c.size() > max_history_) c.pop_front();
    }

    double prob_detection_ = 1.0;
    double prob_survive_ = 1.0;
    double clutter_rate_ = 0.0;
    double clutter_density_ = 0.0;
    std::size_t max_history_ = std::numeric_limits<std::size_t>::max();
};

} // namespace brew::multi_target
