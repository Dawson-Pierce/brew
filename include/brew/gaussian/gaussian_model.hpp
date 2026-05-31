#pragma once

// Notes: Pure data — no sampling, no pdf, no plotting.

#include "brew/shared/base_single_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::models {

/// Gaussian distribution with mean and covariance.
/// Pure parameter holder — merge/filter operations live elsewhere.
// @mex model
// @mex_name Gaussian
// @mex_fields mean:vec, covariance:mat
template <typename T = double, int D = Eigen::Dynamic>
class Gaussian : public BaseSingleModel {
public:
    using Vector = Eigen::Matrix<T, D, 1>;
    using Matrix = Eigen::Matrix<T, D, D>;

    Gaussian() = default;

    inline Gaussian(Vector mean, Matrix covariance)
        : mean_(std::move(mean)), covariance_(std::move(covariance)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<Gaussian<T, D>>(mean_, covariance_);
    }

    [[nodiscard]] inline std::unique_ptr<Gaussian<T, D>> clone_typed() const {
        return std::make_unique<Gaussian<T, D>>(mean_, covariance_);
    }

    // ---- Data access ----
    [[nodiscard]] const Vector& mean() const { return mean_; }
    [[nodiscard]] Vector& mean() { return mean_; }

    [[nodiscard]] const Matrix& covariance() const { return covariance_; }
    [[nodiscard]] Matrix& covariance() { return covariance_; }

private:
    Vector mean_;
    Matrix covariance_;
};

} // namespace brew::models
