#pragma once

#include "dynamics_base.hpp"
#include <unsupported/Eigen/MatrixFunctions>

namespace brew::dynamics {

/// Base class for continuous-time dynamics models defined by the ODE dx/dt = f(x, u).
///
/// Subclasses must provide:
///   f()   — the state derivative
///   F_c() — the continuous-time Jacobian (df/dx)
///   Q_c() — the continuous-time process noise spectral density
///
/// Default implementations:
///   propagate_state()    — 4th-order Runge-Kutta integration
///   get_state_mat()      — matrix exponential: expm(F_c * dt)
///   get_discrete_noise() — Van Loan's method from F_c and Q_c
///
/// Linear subclasses should override propagate_state() and get_state_mat()
/// with analytic solutions for exactness and efficiency.
class ContinuousDynamics : public DynamicsBase {
public:
    /// Continuous-time state derivative: dx/dt = f(x, u).
    [[nodiscard]] virtual Eigen::VectorXd f(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input) const = 0;

    /// Continuous-time Jacobian df/dx (may be state-dependent for nonlinear models).
    [[nodiscard]] virtual Eigen::MatrixXd F_c(
        const Eigen::VectorXd& state = {}) const = 0;

    /// Continuous-time process noise spectral density matrix Q_c.
    [[nodiscard]] virtual Eigen::MatrixXd Q_c() const = 0;

    /// Discrete process noise Q_d(dt) via Van Loan's method.
    /// Call this to initialise filter process noise:
    ///   ekf.set_process_noise(dyn->get_discrete_noise(dt));
    [[nodiscard]] virtual Eigen::MatrixXd get_discrete_noise(
        double dt,
        const Eigen::VectorXd& state = {}) const {

        const Eigen::MatrixXd Fc = F_c(state);
        const Eigen::MatrixXd Qc = Q_c();
        const int n = Fc.rows();

        // Van Loan's method: construct the 2n x 2n auxiliary matrix
        //   Z = [[-Fc,  Qc],
        //        [  0, Fc^T]] * dt
        // then expm(Z) yields F_d and Q_d in its block structure.
        Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(2*n, 2*n);
        Z.topLeftCorner(n, n)     = -Fc;
        Z.topRightCorner(n, n)    =  Qc;
        Z.bottomRightCorner(n, n) =  Fc.transpose();
        Z *= dt;

        const Eigen::MatrixXd eZ = Z.exp();
        const Eigen::MatrixXd Fd = eZ.bottomRightCorner(n, n).transpose();
        return Fd * eZ.topRightCorner(n, n);
    }

    /// State propagation via 4th-order Runge-Kutta.
    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override {

        const Eigen::VectorXd u = input.size() > 0
            ? input : Eigen::VectorXd::Zero(1);

        const auto k1 = f(state,              u);
        const auto k2 = f(state + 0.5*dt*k1,  u);
        const auto k3 = f(state + 0.5*dt*k2,  u);
        const auto k4 = f(state +     dt*k3,   u);
        return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }

    /// Discrete state matrix via matrix exponential: F_d = expm(F_c * dt).
    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const override {
        return (F_c(state) * dt).exp();
    }

    /// No controlled input by default. Override if the model has explicit control inputs.
    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double /*dt*/,
        const Eigen::VectorXd& state = {}) const override {
        return Eigen::MatrixXd::Zero(F_c(state).rows(), 1);
    }
};

} // namespace brew::dynamics
