#pragma once

// Ported from: +BREW/+dynamics/ConstantTurn_2D.m
// Original name: ConstantTurn_2D
// Ported on: 2026-02-07
// Notes: Matches MATLAB linearized position update (no turn-rate position integration).

#include "dynamics_base.hpp"
#include <cmath>

namespace brew::dynamics {

/// 2D constant turn-rate model: state = [x, y, v, theta, omega].
/// Mirrors MATLAB: BREW.dynamics.ConstantTurn_2D
class ConstantTurn2D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<ConstantTurn2D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "v", "theta", "omega"};
    }

    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override {

        Eigen::VectorXd u = input.size() > 0 ? input : Eigen::VectorXd::Zero(2);
        return get_state_mat(dt, state) * state + get_input_mat(dt, state) * u;
    }

    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const override {

        double v = state(2), theta = state(3);

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(5, 5);
        F(0, 2) = std::cos(theta) * dt;
        F(0, 3) = -v * std::sin(theta) * dt;
        F(1, 2) = std::sin(theta) * dt;
        F(1, 3) = v * std::cos(theta) * dt;
        F(3, 4) = dt;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const override {

        const double theta = state(3);
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(5, 2);
        G(0, 0) = dt * std::cos(theta);
        G(1, 0) = dt * std::sin(theta);
        G(2, 0) = 1.0;
        G(4, 1) = 1.0;
        return G;
    }

    [[nodiscard]] Eigen::MatrixXd propagate_extent(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& extent) const override {
        const double omega = state(4);
        const double dtheta = omega * dt;
        const double c = std::cos(dtheta);
        const double s = std::sin(dtheta);
        Eigen::Matrix2d R;
        R << c, -s,
             s,  c;
        return R * extent * R.transpose();
    }
};

} // namespace brew::dynamics
