#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/trajectory_ggiw_orientation.hpp"

namespace brew::filters {

/// EKF for TrajectoryGGIWOrientation distributions.
/// Same trajectory predict/gate math as TrajectoryGGIWEKF; correct adds
/// eigenvector basis tracking that aligns new eigenvectors to the previous
/// basis after each update.
class TrajectoryGGIWOrientationEKF : public Filter<models::TrajectoryGGIWOrientation> {
public:
    TrajectoryGGIWOrientationEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<models::TrajectoryGGIWOrientation>> clone() const override;

    [[nodiscard]] models::TrajectoryGGIWOrientation predict(
        double dt,
        const models::TrajectoryGGIWOrientation& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const models::TrajectoryGGIWOrientation& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const models::TrajectoryGGIWOrientation& predicted) const override;

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
