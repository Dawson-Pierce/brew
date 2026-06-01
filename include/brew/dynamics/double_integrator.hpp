#pragma once

#include "linear_dynamics.hpp"
#include "brew/assert.hpp"

namespace brew::dynamics {

/// Constant-acceleration (double) integrator for 1, 2, or 3 spatial dimensions.
/// State ordering: [pos..., vel..., acc...].
/// Input is an acceleration increment applied each step.
// @mex dynamics
// @mex_name DoubleIntegrator
// @mex_args dims:int
template <typename Scalar = double, int Dspatial = Eigen::Dynamic>
class DoubleIntegrator
    : public LinearDynamics<Scalar, detail::scaled_dim(Dspatial, 3)> {
public:
    static constexpr int StateDim = detail::scaled_dim(Dspatial, 3);
    using Base = LinearDynamics<Scalar, StateDim>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;

    /// Default ctor — only valid when Dspatial is fixed at compile time.
    DoubleIntegrator() : dims_(Dspatial) {
        static_assert(Dspatial != Eigen::Dynamic,
            "Specify spatial dim at runtime via DoubleIntegrator(int)");
        static_assert(Dspatial >= 1 && Dspatial <= 3,
            "DoubleIntegrator: Dspatial must be 1, 2, or 3");
    }

    explicit DoubleIntegrator(int dims) : dims_(dims) {
        BREW_ASSERT(dims >= 1 && dims <= 3,
            "DoubleIntegrator: dims must be 1, 2, or 3");
        if constexpr (Dspatial != Eigen::Dynamic) {
            BREW_ASSERT(dims == Dspatial,
                "DoubleIntegrator: runtime dims does not match template Dspatial");
        }
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, StateDim>> clone() const override {
        return std::make_unique<DoubleIntegrator<Scalar, Dspatial>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        static const std::vector<std::vector<std::string>> names = {
            {"x", "vx", "ax"},
            {"x", "y", "vx", "vy", "ax", "ay"},
            {"x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"}
        };
        return names[dims_ - 1];
    }

    /// F = kron([[1, dt, 0.5*dt^2], [0, 1, dt], [0, 0, 1]], I_dims)
    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& = Vector{}) const override {

        const int n = dims_;
        const auto In =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        Matrix F = Matrix::Identity(3 * n, 3 * n);
        F.block(  0,   n, n, n) = dt * In;
        F.block(  0, 2*n, n, n) = Scalar(0.5) * dt * dt * In;
        F.block(  n, 2*n, n, n) = dt * In;
        return F;
    }

    /// G = kron([[0.5*dt^2], [dt], [1]], I_dims)
    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt,
        const Vector& = Vector{}) const override {

        const int n = dims_;
        const auto In =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        InputMatrix G = InputMatrix::Zero(3 * n, n);
        G.block(  0, 0, n, n) = Scalar(0.5) * dt * dt * In;
        G.block(  n, 0, n, n) = dt * In;
        G.block(2*n, 0, n, n) = In;
        return G;
    }

    /// Linear time-invariant: get_state_mat(dt) is state-independent, so the
    /// transition can be built once and shared across a batch of targets.
    [[nodiscard]] bool is_lti() const override { return true; }

private:
    int dims_;
};

} // namespace brew::dynamics
