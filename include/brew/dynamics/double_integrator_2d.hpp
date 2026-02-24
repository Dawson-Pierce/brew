#pragma once

// Ported from: +BREW/+dynamics/DoubleIntegrator_2D.m
// Original name: DoubleIntegrator_2D
// Ported on: 2026-02-07
// Notes: Matches MATLAB.

#include "dynamics_base.hpp"

namespace brew::dynamics {

/// 2D double integrator: state = [x, y, vx, vy, ax, ay].
/// Mirrors MATLAB: BREW.dynamics.DoubleIntegrator_2D
class DoubleIntegrator2D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<DoubleIntegrator2D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "vx", "vy", "ax", "ay"};
    }

    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override {

        Eigen::VectorXd u = input.size() > 0 ? input : Eigen::VectorXd::Zero(2);
        return get_state_mat(dt) * state + get_input_mat(dt) * u;
    }

    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        double dt2 = 0.5 * dt * dt;
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0, 2) = dt; F(0, 4) = dt2;
        F(1, 3) = dt; F(1, 5) = dt2;
        F(2, 4) = dt;
        F(3, 5) = dt;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        double dt2 = 0.5 * dt * dt;
        Eigen::MatrixXd G(6, 2);
        G << dt2, 0,
             0, dt2,
             dt, 0,
             0, dt,
             1,  0,
             0,  1;
        return G;
    }
};

} // namespace brew::dynamics
