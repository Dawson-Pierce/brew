#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/ggiw.hpp"

namespace brew::filters {

/// Extended Kalman Filter for GGIW distributions.
/// Mirrors MATLAB: BREW.filters.GGIWEKF
class GGIWEKF : public Filter<models::GGIW> {
public:
    GGIWEKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<models::GGIW>> clone() const override;

    [[nodiscard]] models::GGIW predict(
        double dt,
        const models::GGIW& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const models::GGIW& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const models::GGIW& predicted) const override;

    // ---- GGIW-specific parameters ----
    void set_temporal_decay(double eta) { eta_ = eta; }
    void set_forgetting_factor(double tau) { tau_ = tau; }
    void set_scaling_parameter(double rho) { rho_ = rho; }

private:
    double eta_ = 1.0;   ///< Temporal decay (forgetting factor) for gamma parameters
    double tau_ = 1.0;   ///< Exponential decay constant for IW dof
    double rho_ = 1.0;   ///< Scaling parameter for extent in measurement noise
};

} // namespace brew::filters
