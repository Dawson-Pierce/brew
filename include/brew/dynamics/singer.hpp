#pragma once

#include "continuous_dynamics.hpp"
#include <cmath>
#include <stdexcept>

namespace brew::dynamics {

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
class Singer : public ContinuousDynamics {
public:
    Singer(int dims, double beta, double q)
        : dims_(dims), beta_(beta), q_(q) {
        if (dims < 1 || dims > 3)
            throw std::invalid_argument("Singer: dims must be 1, 2, or 3");
        if (beta < 0.0)
            throw std::invalid_argument("Singer: beta must be non-negative");
        if (q < 0.0)
            throw std::invalid_argument("Singer: q must be non-negative");
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<Singer>(*this);
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
    [[nodiscard]] Eigen::VectorXd f(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& /*input*/) const override {
        return F_c() * state;
    }

    /// Continuous Jacobian: F_c = kron([[0,1,0],[0,0,1],[0,0,-beta]], I_dims).
    [[nodiscard]] Eigen::MatrixXd F_c(
        const Eigen::VectorXd& = {}) const override {

        const int n = dims_;
        const auto In = Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd Fc = Eigen::MatrixXd::Zero(3*n, 3*n);
        Fc.block(  0,   n, n, n) =  In;           // pos  <- vel
        Fc.block(  n, 2*n, n, n) =  In;           // vel  <- acc
        Fc.block(2*n, 2*n, n, n) = -beta_ * In;   // acc  <- -beta * acc
        return Fc;
    }

    /// Continuous noise: only acceleration is driven by white noise.
    [[nodiscard]] Eigen::MatrixXd Q_c() const override {
        const int n = dims_;
        Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(3*n, 3*n);
        Qc.block(2*n, 2*n, n, n) = q_ * Eigen::MatrixXd::Identity(n, n);
        return Qc;
    }

    /// Exact analytic discrete state matrix (avoids matrix exponential).
    ///
    /// F_d = kron([[1, dt, (e-1+beta*dt)/beta²],
    ///             [0,  1, (1-e)/beta          ],
    ///             [0,  0,  e                  ]], I_dims)
    ///
    /// where e = exp(-beta * dt).
    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& = {}) const override {

        const int n = dims_;
        const auto In = Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd Fd = Eigen::MatrixXd::Identity(3*n, 3*n);

        if (beta_ < 1e-10) {
            // Limiting case: constant-acceleration (DoubleIntegrator)
            Fd.block(  0,   n, n, n) = dt * In;
            Fd.block(  0, 2*n, n, n) = 0.5 * dt * dt * In;
            Fd.block(  n, 2*n, n, n) = dt * In;
        } else {
            const double e  = std::exp(-beta_ * dt);
            const double bi = 1.0 / beta_;
            Fd.block(  0,   n, n, n) = dt * In;
            Fd.block(  0, 2*n, n, n) = (e - 1.0 + beta_ * dt) * (bi * bi) * In;
            Fd.block(  n, 2*n, n, n) = (1.0 - e) * bi * In;
            Fd.block(2*n, 2*n, n, n) = e * In;
        }
        return Fd;
    }

    /// Exact analytic propagation — linear model, so F_d * x is precise.
    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& /*input*/ = {}) const override {
        return get_state_mat(dt) * state;
    }

    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double /*dt*/,
        const Eigen::VectorXd& = {}) const override {
        return Eigen::MatrixXd::Zero(3 * dims_, 1);
    }

    double beta()              const { return beta_; }
    double spectral_density()  const { return q_; }
    int    dims()              const { return dims_; }

private:
    int    dims_;
    double beta_;
    double q_;
};

} // namespace brew::dynamics
