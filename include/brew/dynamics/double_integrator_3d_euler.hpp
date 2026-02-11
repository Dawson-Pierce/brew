#pragma once

// Ported from: +BREW/+dynamics/DoubleIntegrator_3D_euler.m
// Original name: DoubleIntegrator_3D_euler
// Ported on: 2026-02-07
// Notes: 18-state Euler-angle double integrator.

#include "brew/dynamics/nonlinear_dynamics.hpp"

namespace brew::dynamics {

/// 3D double integrator with Euler angle state:
/// [x,y,z, vx,vy,vz, ax,ay,az, phi,theta,psi, p,q,r, alpha_x,alpha_y,alpha_z].
/// Mirrors MATLAB: BREW.dynamics.DoubleIntegrator_3D_euler
class DoubleIntegrator3DEuler : public NonlinearDynamics {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override;
    [[nodiscard]] std::vector<std::string> state_names() const override;

    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override;

    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const override;

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const override;
};

} // namespace brew::dynamics
