#pragma once

#include "brew/shared/base_single_model.hpp"
#include "brew/usggiw/unit_sphere.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::models {

// @mex model
// @mex_name USGGIW
// @mex_fields alpha:scalar, beta:scalar, direction:vec, angular_velocity:vec, covariance:mat, v:scalar, V:mat
template<typename T = double, int De = Eigen::Dynamic>
class USGGIW : public BaseSingleModel {
public:
    using Vector = Eigen::Matrix<T, 4, 1>;
    using Direction = Eigen::Matrix<T, 3, 1>;
    using Velocity = Eigen::Matrix<T, 2, 1>;
    using Covariance = Eigen::Matrix<T, 4, 4>;
    using Matrix = Covariance;
    using ExtentMatrix = Eigen::Matrix<T, De, De>;

    USGGIW() = default;

    USGGIW(T alpha, T beta, Direction direction, Velocity angular_velocity,
           Covariance covariance, T v, ExtentMatrix V)
        : direction_(brew::usggiw::normalized(direction.template cast<double>()).template cast<T>()),
          angular_velocity_(std::move(angular_velocity)), covariance_(std::move(covariance)),
          alpha_(alpha), beta_(beta), v_(v), V_(std::move(V)) {}

    [[nodiscard]] std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<USGGIW>(*this);
    }
    [[nodiscard]] std::unique_ptr<USGGIW> clone_typed() const { return std::make_unique<USGGIW>(*this); }
    [[nodiscard]] bool is_extended() const override { return true; }

    [[nodiscard]] const Direction& direction() const { return direction_; }
    [[nodiscard]] Direction& direction() { return direction_; }
    [[nodiscard]] const Velocity& angular_velocity() const { return angular_velocity_; }
    [[nodiscard]] Velocity& angular_velocity() { return angular_velocity_; }
    [[nodiscard]] const Covariance& covariance() const { return covariance_; }
    [[nodiscard]] Covariance& covariance() { return covariance_; }
    [[nodiscard]] Vector mean() const {
        Vector state = Vector::Zero();
        state.template tail<2>() = angular_velocity_;
        return state;
    }
    [[nodiscard]] T alpha() const { return alpha_; }
    [[nodiscard]] T& alpha() { return alpha_; }
    [[nodiscard]] T beta() const { return beta_; }
    [[nodiscard]] T& beta() { return beta_; }
    [[nodiscard]] T v() const { return v_; }
    [[nodiscard]] T& v() { return v_; }
    [[nodiscard]] const ExtentMatrix& V() const { return V_; }
    [[nodiscard]] ExtentMatrix& V() { return V_; }
    [[nodiscard]] int extent_dim() const { return static_cast<int>(V_.rows()); }

private:
    Direction direction_ = Direction::UnitX();
    Velocity angular_velocity_ = Velocity::Zero();
    Covariance covariance_ = Covariance::Identity();
    T alpha_ = T(0);
    T beta_ = T(0);
    T v_ = T(0);
    ExtentMatrix V_;
};

}  // namespace brew::models
