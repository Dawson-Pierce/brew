#pragma once

// Ported from: +BREW/+dynamics/Integrator_1D.m
// Original name: Integrator_1D
// Ported on: 2026-02-07
// Notes: Matches MATLAB state matrix (velocity row zeroed).

#include "dynamics_base.hpp"

namespace brew::dynamics {

/// 1D constant-velocity integrator: state = [x, vx].
/// Mirrors MATLAB: BREW.dynamics.Integrator_1D
class Integrator1D : public DynamicsBase {
public:
    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<Integrator1D>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "vx"};
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

        Eigen::Matrix2d F;
        F << 1, dt,
             0, 0;
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::MatrixXd G(2, 1);
        G << dt, 1;
        return G;
    }
};

} // namespace brew::dynamics
