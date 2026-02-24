#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/gaussian.hpp"

namespace brew::filters {

/// Extended Kalman Filter for Gaussian distributions.
/// Mirrors MATLAB: BREW.filters.EKF
class EKF : public Filter<models::Gaussian> {
public:
    EKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<models::Gaussian>> clone() const override;

    [[nodiscard]] models::Gaussian predict(
        double dt,
        const models::Gaussian& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const models::Gaussian& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const models::Gaussian& predicted) const override;
};

} // namespace brew::filters
