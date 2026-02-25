#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/ggiw_orientation.hpp"

namespace brew::filters {

/// EKF for GGIWOrientation distributions.
/// Same predict/gate math as GGIWEKF; correct adds eigenvector basis tracking
/// that aligns new eigenvectors to the previous basis after each update.
class GGIWOrientationEKF : public Filter<models::GGIWOrientation> {
public:
    GGIWOrientationEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<models::GGIWOrientation>> clone() const override;

    [[nodiscard]] models::GGIWOrientation predict(
        double dt,
        const models::GGIWOrientation& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const models::GGIWOrientation& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const models::GGIWOrientation& predicted) const override;

    // ---- GGIW-specific parameters ----
    void set_temporal_decay(double eta) { eta_ = eta; }
    void set_forgetting_factor(double tau) { tau_ = tau; }
    void set_scaling_parameter(double rho) { rho_ = rho; }

private:
    double eta_ = 1.0;   ///< Temporal decay for gamma parameters
    double tau_ = 1.0;   ///< Exponential decay constant for IW dof
    double rho_ = 1.0;   ///< Scaling parameter for extent in measurement noise
};

} // namespace brew::filters
