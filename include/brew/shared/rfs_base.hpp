#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <memory>

namespace brew::multi_target {

class RFSBase {
public:
    virtual ~RFSBase() = default;

    [[nodiscard]] virtual std::unique_ptr<RFSBase> clone() const = 0;

    void set_prob_detection(double pd) { prob_detection_ = pd; }
    void set_prob_survive(double ps) { prob_survive_ = ps; }
    void set_clutter_rate(double rate) { clutter_rate_ = rate; }
    void set_clutter_density(double density) { clutter_density_ = density; }

    void set_max_history(std::size_t n) { max_history_ = n; }

    [[nodiscard]] double prob_detection() const { return prob_detection_; }
    [[nodiscard]] double prob_survive() const { return prob_survive_; }
    [[nodiscard]] double clutter_rate() const { return clutter_rate_; }
    [[nodiscard]] double clutter_density() const { return clutter_density_; }
    [[nodiscard]] std::size_t max_history() const { return max_history_; }

    virtual void predict(int timestep, double dt) = 0;
    virtual void correct(const Eigen::MatrixXd& measurements) = 0;
    virtual void cleanup() = 0;

protected:

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

}
