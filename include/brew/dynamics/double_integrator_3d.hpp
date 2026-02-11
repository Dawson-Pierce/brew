#pragma once

// Ported from: +BREW/+dynamics/DoubleIntegrator_3D.m
// Original name: DoubleIntegrator_3D
// Ported on: 2026-02-07
// Notes: Input matrix matches MATLAB (row 3 mirrors row 1).

#include "dynamics_base.hpp"

namespace brew::dynamics {

/// 3D double integrator: state = [x, y, z, vx, vy, vz, ax, ay, az].
/// Mirrors MATLAB: BREW.dynamics.DoubleIntegrator_3D
class DoubleIntegrator3D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<DoubleIntegrator3D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"};
    }

    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override {

        Eigen::VectorXd u = input.size() > 0 ? input : Eigen::VectorXd::Zero(3);
        return get_state_mat(dt) * state + get_input_mat(dt) * u;
    }

    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        double dt2 = 0.5 * dt * dt;
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(9, 9);
        F(0, 3) = dt; F(0, 6) = dt2;
        F(1, 4) = dt; F(1, 7) = dt2;
        F(2, 5) = dt; F(2, 8) = dt2;
        F(3, 6) = dt;
        F(4, 7) = dt;
        F(5, 8) = dt;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        double dt2 = 0.5 * dt * dt;
        Eigen::MatrixXd G(9, 3);
        G << dt2, 0, 0,
             0, dt2, 0,
             dt2, 0, 0,
             dt, 0, 0,
             0, dt, 0,
             0, 0, dt,
             1,  0, 0,
             0,  1, 0,
             0,  0, 1;
        return G;
    }
};

} // namespace brew::dynamics
