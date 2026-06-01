#pragma once

#include "linear_dynamics.hpp"
#include "brew/assert.hpp"

namespace brew::dynamics {

/// Constant-velocity integrator for 1, 2, or 3 spatial dimensions.
/// State ordering: [pos..., vel...].
/// Input is a velocity increment applied each step.
// @mex dynamics
// @mex_name SingleIntegrator
// @mex_args dims:int
template <typename Scalar = double, int Dspatial = Eigen::Dynamic>
class SingleIntegrator
    : public LinearDynamics<Scalar, detail::scaled_dim(Dspatial, 2)> {
public:
    static constexpr int StateDim = detail::scaled_dim(Dspatial, 2);
    using Base = LinearDynamics<Scalar, StateDim>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;

    /// Default ctor — only valid when Dspatial is fixed at compile time.
    SingleIntegrator() : dims_(Dspatial) {
        static_assert(Dspatial != Eigen::Dynamic,
            "Specify spatial dim at runtime via SingleIntegrator(int)");
        static_assert(Dspatial >= 1 && Dspatial <= 3,
            "SingleIntegrator: Dspatial must be 1, 2, or 3");
    }

    explicit SingleIntegrator(int dims) : dims_(dims) {
        BREW_ASSERT(dims >= 1 && dims <= 3,
            "SingleIntegrator: dims must be 1, 2, or 3");
        if constexpr (Dspatial != Eigen::Dynamic) {
            BREW_ASSERT(dims == Dspatial,
                "SingleIntegrator: runtime dims does not match template Dspatial");
        }
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, StateDim>> clone() const override {
        return std::make_unique<SingleIntegrator<Scalar, Dspatial>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        static const std::vector<std::vector<std::string>> names = {
            {"x", "vx"},
            {"x", "y", "vx", "vy"},
            {"x", "y", "z", "vx", "vy", "vz"}
        };
        return names[dims_ - 1];
    }

    /// F = kron([[1, dt], [0, 1]], I_dims)
    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& = Vector{}) const override {

        const int n = dims_;
        Matrix F = Matrix::Identity(2 * n, 2 * n);
        F.block(0, n, n, n) =
            dt * Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        return F;
    }

    /// G = kron([[dt], [1]], I_dims)
    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt,
        const Vector& = Vector{}) const override {

        const int n = dims_;
        InputMatrix G = InputMatrix::Zero(2 * n, n);
        G.block(0, 0, n, n) =
            dt * Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        G.block(n, 0, n, n) =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        return G;
    }

    /// Linear time-invariant: get_state_mat(dt) is state-independent, so the
    /// transition can be built once and shared across a batch of targets.
    [[nodiscard]] bool is_lti() const override { return true; }

private:
    int dims_;
};

} // namespace brew::dynamics
