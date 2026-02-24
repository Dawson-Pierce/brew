#pragma once

#include <Eigen/Dense>
#include <memory>

namespace brew::multi_target {

/// Abstract base class for Random Finite Set filters.
/// Mirrors MATLAB: BREW.multi_target.RFSBase
class RFSBase {
public:
    virtual ~RFSBase() = default;

    [[nodiscard]] virtual std::unique_ptr<RFSBase> clone() const = 0;

    // ---- Configuration ----

    void set_prob_detection(double pd) { prob_detection_ = pd; }
    void set_prob_survive(double ps) { prob_survive_ = ps; }
    void set_clutter_rate(double rate) { clutter_rate_ = rate; }
    void set_clutter_density(double density) { clutter_density_ = density; }

    [[nodiscard]] double prob_detection() const { return prob_detection_; }
    [[nodiscard]] double prob_survive() const { return prob_survive_; }
    [[nodiscard]] double clutter_rate() const { return clutter_rate_; }
    [[nodiscard]] double clutter_density() const { return clutter_density_; }

    // ---- Abstract interface ----

    virtual void predict(int timestep, double dt) = 0;
    virtual void correct(const Eigen::MatrixXd& measurements) = 0;
    virtual void cleanup() = 0;

protected:
    double prob_detection_ = 1.0;
    double prob_survive_ = 1.0;
    double clutter_rate_ = 0.0;
    double clutter_density_ = 0.0;
};

} // namespace brew::multi_target
