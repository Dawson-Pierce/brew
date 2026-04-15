#pragma once

#include "brew/core/dynamics/dynamics_base.hpp"
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
template <typename Scalar = double, int D = Eigen::Dynamic>
class ContinuousDynamics : public DynamicsBase<Scalar, D> {
public:
    using Base = DynamicsBase<Scalar, D>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;

    [[nodiscard]] virtual Vector f(
        const Vector& state,
        const InputVector& input) const = 0;

    [[nodiscard]] virtual Matrix F_c(const Vector& state = Vector{}) const = 0;

    [[nodiscard]] virtual Matrix Q_c() const = 0;

    [[nodiscard]] virtual Matrix get_discrete_noise(
        Scalar dt,
        const Vector& state = Vector{}) const {

        const Matrix Fc = F_c(state);
        const Matrix Qc = Q_c();
        const int n = static_cast<int>(Fc.rows());

        // Van Loan's 2n×2n auxiliary matrix — stays dynamic even for fixed D
        // because the block expansion is 2n on each side.
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Z =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(2*n, 2*n);
        Z.topLeftCorner(n, n)     = -Fc;
        Z.topRightCorner(n, n)    =  Qc;
        Z.bottomRightCorner(n, n) =  Fc.transpose();
        Z *= dt;

        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eZ = Z.exp();
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Fd =
            eZ.bottomRightCorner(n, n).transpose();
        return Fd * eZ.topRightCorner(n, n);
    }

    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& input = InputVector{}) const override {

        const InputVector u = input.size() > 0
            ? input : InputVector::Zero(1);

        const auto k1 = f(state,                       u);
        const auto k2 = f((state + Scalar(0.5)*dt*k1).eval(), u);
        const auto k3 = f((state + Scalar(0.5)*dt*k2).eval(), u);
        const auto k4 = f((state + dt*k3).eval(),             u);
        return state + (dt / Scalar(6)) * (k1 + Scalar(2)*k2 + Scalar(2)*k3 + k4);
    }

    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& state = Vector{}) const override {
        return (F_c(state) * dt).exp();
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar /*dt*/,
        const Vector& state = Vector{}) const override {
        return InputMatrix::Zero(F_c(state).rows(), 1);
    }
};

} // namespace brew::dynamics
