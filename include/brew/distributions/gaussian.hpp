#pragma once

// Ported from: +BREW/+distributions/Gaussian.m
// Original name: Gaussian
// Ported on: 2026-02-07
// Notes: Pure data — no sampling, no pdf, no plotting.

#include "brew/distributions/base_single_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::distributions {

/// Gaussian distribution with mean and covariance.
/// Pure parameter holder — merge/filter operations live elsewhere.
class Gaussian : public BaseSingleModel {
public:
    Gaussian() = default;

    inline Gaussian(Eigen::VectorXd mean, Eigen::MatrixXd covariance)
        : mean_(std::move(mean)), covariance_(std::move(covariance)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<Gaussian>(mean_, covariance_);
    }

    [[nodiscard]] inline std::unique_ptr<Gaussian> clone_typed() const {
        return std::make_unique<Gaussian>(mean_, covariance_);
    }

    // ---- Data access ----
    [[nodiscard]] const Eigen::VectorXd& mean() const { return mean_; }
    [[nodiscard]] Eigen::VectorXd& mean() { return mean_; }

    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return covariance_; }
    [[nodiscard]] Eigen::MatrixXd& covariance() { return covariance_; }

private:
    Eigen::VectorXd mean_;
    Eigen::MatrixXd covariance_;
};

} // namespace brew::distributions
