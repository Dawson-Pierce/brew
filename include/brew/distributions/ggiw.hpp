#pragma once

// Ported from: +BREW/+distributions/GGIW.m
// Original name: GGIW
// Ported on: 2026-02-07
// Notes: Pure data — no sampling, no plotting.

#include "brew/distributions/base_single_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::distributions {

/// Gamma-Gaussian-Inverse-Wishart distribution.
/// Pure parameter holder for kinematic state + measurement rate + extent.
class GGIW : public BaseSingleModel {
public:
    GGIW() = default;

    inline GGIW(Eigen::VectorXd mean, Eigen::MatrixXd covariance,
                double alpha, double beta,
                double v, Eigen::MatrixXd V)
        : mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          alpha_(alpha), beta_(beta),
          v_(v), V_(std::move(V)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<GGIW>(mean_, covariance_, alpha_, beta_, v_, V_);
    }

    [[nodiscard]] inline std::unique_ptr<GGIW> clone_typed() const {
        return std::make_unique<GGIW>(mean_, covariance_, alpha_, beta_, v_, V_);
    }

    // ---- Kinematic state ----
    [[nodiscard]] const Eigen::VectorXd& mean() const { return mean_; }
    [[nodiscard]] Eigen::VectorXd& mean() { return mean_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return covariance_; }
    [[nodiscard]] Eigen::MatrixXd& covariance() { return covariance_; }

    // ---- Gamma (measurement rate) ----
    [[nodiscard]] double alpha() const { return alpha_; }
    [[nodiscard]] double& alpha() { return alpha_; }
    [[nodiscard]] double beta() const { return beta_; }
    [[nodiscard]] double& beta() { return beta_; }

    // ---- Inverse-Wishart (extent) ----
    [[nodiscard]] double v() const { return v_; }
    [[nodiscard]] double& v() { return v_; }
    [[nodiscard]] const Eigen::MatrixXd& V() const { return V_; }
    [[nodiscard]] Eigen::MatrixXd& V() { return V_; }
    [[nodiscard]] int extent_dim() const { return static_cast<int>(V_.rows()); }

private:
    Eigen::VectorXd mean_;
    Eigen::MatrixXd covariance_;
    double alpha_ = 0.0;
    double beta_ = 0.0;
    double v_ = 0.0;
    Eigen::MatrixXd V_;
};

} // namespace brew::distributions
