#pragma once

#include "brew/dynamics/nonlinear_dynamics.hpp"
#include <stdexcept>
#include <cmath>

namespace brew::dynamics {

/// Constant body-frame velocity integrator for 2D or 3D.
/// State ordering: [pos..., vel_body...].
/// Body-frame velocities are rotated into the world frame for position update:
///   p_{k+1} = p_k + R * v_body * dt
///   v_body_{k+1} = v_body_k
///
/// The Jacobian (F) and input matrix (G) depend on the rotation matrix,
/// so use the rotation-aware overloads.
class BodyFrameSingleIntegrator : public NonlinearDynamics {
public:
    explicit BodyFrameSingleIntegrator(int dims) : dims_(dims) {
        if (dims < 2 || dims > 3)
            throw std::invalid_argument("BodyFrameSingleIntegrator: dims must be 2 or 3");
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<BodyFrameSingleIntegrator>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        if (dims_ == 2)
            return {"x", "y", "v_fwd", "v_lat"};
        return {"x", "y", "z", "v_fwd", "v_lat", "v_up"};
    }

    /// Rotation-independent propagation (falls back to world-frame, identity rotation).
    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& /*input*/ = {}) const override {

        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dims_, dims_);
        return propagate_state(dt, state, R);
    }

    /// Propagate with rotation: p += R * v_body * dt, v_body unchanged.
    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& rotation,
        const Eigen::VectorXd& /*input*/ = {}) const override {

        const int n = dims_;
        Eigen::VectorXd next = state;
        Eigen::VectorXd v_body = state.tail(n);
        next.head(n) += dt * rotation.topLeftCorner(n, n) * v_body;
        return next;
    }

    /// Jacobian F without rotation (identity rotation assumed).
    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dims_, dims_);
        return get_state_mat_impl(dt, R);
    }

    /// Jacobian F with rotation: d(p_{k+1})/d(v_body) = R * dt.
    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& /*state*/,
        const Eigen::MatrixXd& rotation) const override {

        return get_state_mat_impl(dt, rotation);
    }

    /// Input matrix G without rotation.
    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/ = {}) const override {

        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dims_, dims_);
        return get_input_mat_impl(dt, R);
    }

    /// Input matrix G with rotation.
    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& /*state*/,
        const Eigen::MatrixXd& rotation) const override {

        return get_input_mat_impl(dt, rotation);
    }

private:
    int dims_;

    [[nodiscard]] Eigen::MatrixXd get_state_mat_impl(
        double dt, const Eigen::MatrixXd& rotation) const {

        const int n = dims_;
        // F = [I, R*dt; 0, I]
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2 * n, 2 * n);
        F.block(0, n, n, n) = dt * rotation.topLeftCorner(n, n);
        return F;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat_impl(
        double dt, const Eigen::MatrixXd& rotation) const {

        const int n = dims_;
        // G = [R*dt; I]  — process noise enters in body frame
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(2 * n, n);
        G.block(0, 0, n, n) = dt * rotation.topLeftCorner(n, n);
        G.block(n, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
        return G;
    }
};

} // namespace brew::dynamics
