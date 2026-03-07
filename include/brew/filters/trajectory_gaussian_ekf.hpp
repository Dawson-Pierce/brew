#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/gaussian.hpp"

namespace brew::filters {

/// EKF for Gaussian trajectory distributions.

class TrajectoryGaussianEKF : public Filter<models::Trajectory<models::Gaussian>> {
public:
    using TrajectoryType = models::Trajectory<models::Gaussian>;

    TrajectoryGaussianEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<TrajectoryType>> clone() const override;

    [[nodiscard]] TrajectoryType predict(
        double dt,
        const TrajectoryType& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const TrajectoryType& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const TrajectoryType& predicted) const override;

    void set_window_size(int L) { l_window_ = L; }

private:
    int l_window_ = 50;
};

} // namespace brew::filters
