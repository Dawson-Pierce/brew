#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/ggiw_orientation.hpp"

namespace brew::filters {

/// EKF for Trajectory<GGIWOrientation> distributions.
/// Same trajectory predict/gate math as TrajectoryGGIWEKF; correct adds
/// eigenvector basis tracking that aligns new eigenvectors to the previous
/// basis after each update.
class TrajectoryGGIWOrientationEKF : public Filter<models::Trajectory<models::GGIWOrientation>> {
public:
    using TrajectoryType = models::Trajectory<models::GGIWOrientation>;

    TrajectoryGGIWOrientationEKF() = default;

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

    void set_temporal_decay(double eta) { eta_ = eta; }
    void set_forgetting_factor(double tau) { tau_ = tau; }
    void set_scaling_parameter(double rho) { rho_ = rho; }
    void set_window_size(int L) { l_window_ = L; }

private:
    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;
    int l_window_ = 50;
};

} // namespace brew::filters
