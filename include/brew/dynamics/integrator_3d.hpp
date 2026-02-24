#pragma once

// Ported from: +BREW/+dynamics/Integrator_3D.m
// Original name: Integrator_3D
// Ported on: 2026-02-07
// Notes: Matches MATLAB.

#include "dynamics_base.hpp"

namespace brew::dynamics {

/// 3D constant-velocity integrator: state = [x, y, z, vx, vy, vz].
/// Mirrors MATLAB: BREW.dynamics.Integrator_3D
class Integrator3D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<Integrator3D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "z", "vx", "vy", "vz"};
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

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::MatrixXd G(6, 3);
        G << dt, 0, 0,
             0, dt, 0,
             0, 0, dt,
             1,  0, 0,
             0,  1, 0,
             0,  0, 1;
        return G;
    }
};

} // namespace brew::dynamics
