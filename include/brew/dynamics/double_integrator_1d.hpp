#pragma once

// Ported from: +BREW/+dynamics/DoubleIntegrator_1D.m
// Original name: DoubleIntegrator_1D
// Ported on: 2026-02-07
// Notes: Matches MATLAB state matrix.

#include "dynamics_base.hpp"

namespace brew::dynamics {

/// 1D double integrator: state = [x, vx, ax].
/// Mirrors MATLAB: BREW.dynamics.DoubleIntegrator_1D
class DoubleIntegrator1D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<DoubleIntegrator1D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "vx", "ax"};
    }

    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override {

        Eigen::VectorXd u = input.size() > 0 ? input : Eigen::VectorXd::Zero(1);
        return get_state_mat(dt) * state + get_input_mat(dt) * u;
    }

    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::Matrix3d F;
        F << 1, dt, 0.5*dt*dt,
             0, 0,  dt,
             0, 0,  0;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::MatrixXd G(3, 1);
        G << 0.5*dt*dt, dt, 1;
        return G;
    }
};

} // namespace brew::dynamics
