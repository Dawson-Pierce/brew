#pragma once

#include "continuous_dynamics.hpp"
#include "brew/assert.hpp"
#include <cmath>

namespace brew::dynamics {

// @mex dynamics
// @mex_name Singer
// @mex_args dims:int, beta:double, q:double
/// Singer auto-regressive acceleration model for 1, 2, or 3 spatial dimensions.
///
/// State ordering: [pos..., vel..., acc...] — identical to DoubleIntegrator.
///
/// Continuous-time ODE per axis:
///   d(pos)/dt =  vel
///   d(vel)/dt =  acc
///   d(acc)/dt = -beta * acc + w   (w ~ white noise with spectral density q)
///
/// beta  — maneuver rate (1/s). Inverse of maneuver time constant.
///          Typical values: 1/60 (slow), 1/20 (moderate), 1/5 (aggressive).
/// q     — acceleration spectral density (m²/s³ per axis).
///
/// As beta → 0, the model degenerates to the constant-acceleration DoubleIntegrator.
/// Use get_discrete_noise(dt) from the base class to obtain the process noise matrix Q_d.
template <typename Scalar = double, int Dspatial = Eigen::Dynamic>
class Singer : public ContinuousDynamics<Scalar, detail::scaled_dim(Dspatial, 3)> {
public:
    static constexpr int StateDim = detail::scaled_dim(Dspatial, 3);
    using Base = ContinuousDynamics<Scalar, StateDim>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;

    /// Fixed-Dspatial ctor: spatial dim is a compile-time template parameter.
    Singer(Scalar beta, Scalar q)
        : dims_(Dspatial), beta_(beta), q_(q) {
        static_assert(Dspatial != Eigen::Dynamic,
            "Specify spatial dim at runtime via Singer(int, Scalar, Scalar)");
        static_assert(Dspatial >= 1 && Dspatial <= 3,
            "Singer: Dspatial must be 1, 2, or 3");
        BREW_ASSERT(beta >= Scalar(0), "Singer: beta must be non-negative");
        BREW_ASSERT(q >= Scalar(0), "Singer: q must be non-negative");
    }

    /// Dynamic-Dspatial ctor.
    Singer(int dims, Scalar beta, Scalar q)
        : dims_(dims), beta_(beta), q_(q) {
        BREW_ASSERT(dims >= 1 && dims <= 3, "Singer: dims must be 1, 2, or 3");
        BREW_ASSERT(beta >= Scalar(0), "Singer: beta must be non-negative");
        BREW_ASSERT(q >= Scalar(0), "Singer: q must be non-negative");
        if constexpr (Dspatial != Eigen::Dynamic) {
            BREW_ASSERT(dims == Dspatial,
                "Singer: runtime dims does not match template Dspatial");
        }
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, StateDim>> clone() const override {
        return std::make_unique<Singer<Scalar, Dspatial>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        static const std::vector<std::vector<std::string>> names = {
            {"x", "vx", "ax"},
            {"x", "y", "vx", "vy", "ax", "ay"},
            {"x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"}
        };
        return names[dims_ - 1];
    }

    /// dx/dt = F_c * x  (linear model, input unused).
    [[nodiscard]] Vector f(
        const Vector& state,
        const InputVector& /*input*/) const override {
        return F_c() * state;
    }

    /// Continuous Jacobian: F_c = kron([[0,1,0],[0,0,1],[0,0,-beta]], I_dims).
    [[nodiscard]] Matrix F_c(
        const Vector& = Vector{}) const override {

        const int n = dims_;
        const auto In =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        Matrix Fc = Matrix::Zero(3 * n, 3 * n);
        Fc.block(  0,   n, n, n) =  In;           // pos  <- vel
        Fc.block(  n, 2*n, n, n) =  In;           // vel  <- acc
        Fc.block(2*n, 2*n, n, n) = -beta_ * In;   // acc  <- -beta * acc
        return Fc;
    }

    /// Continuous noise: only acceleration is driven by white noise.
    [[nodiscard]] Matrix Q_c() const override {
        const int n = dims_;
        Matrix Qc = Matrix::Zero(3 * n, 3 * n);
        Qc.block(2*n, 2*n, n, n) = q_ *
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        return Qc;
    }

    /// Exact analytic discrete state matrix (avoids matrix exponential).
    ///
    /// F_d = kron([[1, dt, (e-1+beta*dt)/beta²],
    ///             [0,  1, (1-e)/beta          ],
    ///             [0,  0,  e                  ]], I_dims)
    ///
    /// where e = exp(-beta * dt).
    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& = Vector{}) const override {

        const int n = dims_;
        const auto In =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        Matrix Fd = Matrix::Identity(3 * n, 3 * n);

        if (beta_ < Scalar(1e-10)) {
            // Limiting case: constant-acceleration (DoubleIntegrator)
            Fd.block(  0,   n, n, n) = dt * In;
            Fd.block(  0, 2*n, n, n) = Scalar(0.5) * dt * dt * In;
            Fd.block(  n, 2*n, n, n) = dt * In;
        } else {
            const Scalar e  = std::exp(-beta_ * dt);
            const Scalar bi = Scalar(1) / beta_;
            Fd.block(  0,   n, n, n) = dt * In;
            Fd.block(  0, 2*n, n, n) = (e - Scalar(1) + beta_ * dt) * (bi * bi) * In;
            Fd.block(  n, 2*n, n, n) = (Scalar(1) - e) * bi * In;
            Fd.block(2*n, 2*n, n, n) = e * In;
        }
        return Fd;
    }

    /// Exact analytic propagation — linear model, so F_d * x is precise.
    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& /*input*/ = InputVector{}) const override {
        return get_state_mat(dt) * state;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar /*dt*/,
        const Vector& = Vector{}) const override {
        return InputMatrix::Zero(3 * dims_, 1);
    }

    /// Linear time-invariant: get_state_mat(dt) (= F_d) is state-independent.
    [[nodiscard]] bool is_lti() const override { return true; }

    Scalar beta()              const { return beta_; }
    Scalar spectral_density()  const { return q_; }
    int    dims()              const { return dims_; }

private:
    int    dims_;
    Scalar beta_;
    Scalar q_;
};

} // namespace brew::dynamics
