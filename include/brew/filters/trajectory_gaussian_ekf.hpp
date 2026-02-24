#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/trajectory_gaussian.hpp"

namespace brew::filters {

/// EKF for Gaussian trajectory distributions.
/// Mirrors MATLAB: BREW.filters.TrajectoryGaussianEKF
class TrajectoryGaussianEKF : public Filter<models::TrajectoryGaussian> {
public:
    TrajectoryGaussianEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<models::TrajectoryGaussian>> clone() const override;

    [[nodiscard]] models::TrajectoryGaussian predict(
        double dt,
        const models::TrajectoryGaussian& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const models::TrajectoryGaussian& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const models::TrajectoryGaussian& predicted) const override;

    void set_window_size(int L) { l_window_ = L; }

private:
    int l_window_ = 50;
};

} // namespace brew::filters
