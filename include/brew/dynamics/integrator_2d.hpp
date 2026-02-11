#pragma once

// Ported from: +BREW/+dynamics/Integrator_2D.m
// Original name: Integrator_2D
// Ported on: 2026-02-07
// Notes: Matches MATLAB.

#include "dynamics_base.hpp"

namespace brew::dynamics {

/// 2D constant-velocity integrator: state = [x, y, vx, vy].
/// Mirrors MATLAB: BREW.dynamics.Integrator_2D
class Integrator2D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<Integrator2D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "vx", "vy"};
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

        Eigen::Matrix4d F;
        F << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1,  0,
             0, 0, 0,  1;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::MatrixXd G(4, 2);
        G << dt, 0,
             0, dt,
             1,  0,
             0,  1;
        return G;
    }
};

} // namespace brew::dynamics
