#pragma once

#include "brew/filters/filter.hpp"
#include "brew/distributions/trajectory_gaussian.hpp"

namespace brew::filters {

/// EKF for Gaussian trajectory distributions.
/// Mirrors MATLAB: BREW.filters.TrajectoryGaussianEKF
class TrajectoryGaussianEKF : public Filter<distributions::TrajectoryGaussian> {
public:
    TrajectoryGaussianEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<distributions::TrajectoryGaussian>> clone() const override;

    [[nodiscard]] distributions::TrajectoryGaussian predict(
        double dt,
        const distributions::TrajectoryGaussian& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const distributions::TrajectoryGaussian& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const distributions::TrajectoryGaussian& predicted) const override;

    void set_window_size(int L) { l_window_ = L; }

private:
    int l_window_ = 50;
};

} // namespace brew::filters
