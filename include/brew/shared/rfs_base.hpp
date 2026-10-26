#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace brew::multi_target {

class RFSBase {
public:
    virtual ~RFSBase() = default;

    [[nodiscard]] virtual std::unique_ptr<RFSBase> clone() const = 0;

    void set_prob_detection(double pd) {
        require_probability(pd, "Detection probability");
        prob_detection_ = pd;
    }
    void set_prob_survive(double ps) {
        require_probability(ps, "Survival probability");
        prob_survive_ = ps;
    }
    void set_clutter_rate(double rate) {
        require_nonnegative(rate, "Clutter rate");
        clutter_rate_ = rate;
    }
    void set_clutter_density(double density) {
        require_nonnegative(density, "Clutter density");
        clutter_density_ = density;
    }

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

    static void require_probability(double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            throw std::invalid_argument(std::string(name) + " must lie in [0, 1].");
        }
    }

    static void require_nonnegative(double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and nonnegative.");
        }
    }

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
