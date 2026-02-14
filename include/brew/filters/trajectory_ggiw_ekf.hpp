#pragma once

#include "brew/filters/filter.hpp"
#include "brew/distributions/trajectory_ggiw.hpp"

namespace brew::filters {

/// EKF for GGIW trajectory distributions.
/// Mirrors MATLAB: BREW.filters.TrajectoryGGIWEKF
class TrajectoryGGIWEKF : public Filter<distributions::TrajectoryGGIW> {
public:
    TrajectoryGGIWEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<distributions::TrajectoryGGIW>> clone() const override;

    [[nodiscard]] distributions::TrajectoryGGIW predict(
        double dt,
        const distributions::TrajectoryGGIW& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const distributions::TrajectoryGGIW& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const distributions::TrajectoryGGIW& predicted) const override;

    void set_temporal_decay(double eta) { eta_ = eta; }
    void set_forgetting_factor(double tau) { tau_ = tau; }
    void set_scaling_parameter(double rho) { rho_ = rho; }
    void set_window_size(int L) { l_window_ = L; }

private:
    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;   ///< Scaling parameter for extent in measurement noise
    int l_window_ = 50;
};

} // namespace brew::filters
