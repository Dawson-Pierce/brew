#pragma once

#include "brew/filters/filter.hpp"
#include "brew/distributions/gaussian.hpp"

namespace brew::filters {

/// Extended Kalman Filter for Gaussian distributions.
/// Mirrors MATLAB: BREW.filters.EKF
class EKF : public Filter<distributions::Gaussian> {
public:
    EKF() = default;

    [[nodiscard]] std::unique_ptr<Filter<distributions::Gaussian>> clone() const override;

    [[nodiscard]] distributions::Gaussian predict(
        double dt,
        const distributions::Gaussian& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const distributions::Gaussian& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const distributions::Gaussian& predicted) const override;
};

} // namespace brew::filters
