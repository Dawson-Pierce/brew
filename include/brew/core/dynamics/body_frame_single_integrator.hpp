#pragma once

#include "brew/core/dynamics/nonlinear_dynamics.hpp"
#include "brew/core/assert.hpp"
#include <cmath>

namespace brew::dynamics {

// @mex dynamics
// @mex_name BodyFrameSingleIntegrator
// @mex_args dims:int
/// Constant body-frame velocity integrator for 2D or 3D.
/// State ordering: [pos..., vel_body...].
/// Body-frame velocities are rotated into the world frame for position update:
///   p_{k+1} = p_k + R * v_body * dt
///   v_body_{k+1} = v_body_k
///
/// The Jacobian (F) and input matrix (G) depend on the rotation matrix,
/// so use the rotation-aware overloads.
template <typename Scalar = double, int Dspatial = Eigen::Dynamic>
class BodyFrameSingleIntegrator
    : public NonlinearDynamics<Scalar, detail::scaled_dim(Dspatial, 2)> {
public:
    static constexpr int StateDim = detail::scaled_dim(Dspatial, 2);
    using Base = NonlinearDynamics<Scalar, StateDim>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;
    using RotationMatrix = typename Base::RotationMatrix;

    /// Default ctor — only valid when Dspatial is fixed at compile time.
    BodyFrameSingleIntegrator() : dims_(Dspatial) {
        static_assert(Dspatial != Eigen::Dynamic,
            "Specify spatial dim at runtime via BodyFrameSingleIntegrator(int)");
        static_assert(Dspatial == 2 || Dspatial == 3,
            "BodyFrameSingleIntegrator: Dspatial must be 2 or 3");
    }

    explicit BodyFrameSingleIntegrator(int dims) : dims_(dims) {
        BREW_ASSERT(dims >= 2 && dims <= 3,
            "BodyFrameSingleIntegrator: dims must be 2 or 3");
        if constexpr (Dspatial != Eigen::Dynamic) {
            BREW_ASSERT(dims == Dspatial,
                "BodyFrameSingleIntegrator: runtime dims does not match template Dspatial");
        }
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, StateDim>> clone() const override {
        return std::make_unique<BodyFrameSingleIntegrator<Scalar, Dspatial>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        if (dims_ == 2)
            return {"x", "y", "v_fwd", "v_lat"};
        return {"x", "y", "z", "v_fwd", "v_lat", "v_up"};
    }

    /// Rotation-independent propagation (falls back to world-frame, identity rotation).
    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& /*input*/ = InputVector{}) const override {

        RotationMatrix R = RotationMatrix::Identity(dims_, dims_);
        return propagate_state(dt, state, R);
    }

    /// Propagate with rotation: p += R * v_body * dt, v_body unchanged.
    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& rotation,
        const InputVector& /*input*/ = InputVector{}) const override {

        const int n = dims_;
        Vector next = state;
        Vector v_body = state.tail(n);
        next.head(n) += dt * rotation.topLeftCorner(n, n) * v_body;
        return next;
    }

    /// Jacobian F without rotation (identity rotation assumed).
    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& /*state*/ = Vector{}) const override {

        RotationMatrix R = RotationMatrix::Identity(dims_, dims_);
        return get_state_mat_impl(dt, R);
    }

    /// Jacobian F with rotation: d(p_{k+1})/d(v_body) = R * dt.
    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& /*state*/,
        const RotationMatrix& rotation) const override {

        return get_state_mat_impl(dt, rotation);
    }

    /// Input matrix G without rotation.
    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt,
        const Vector& /*state*/ = Vector{}) const override {

        RotationMatrix R = RotationMatrix::Identity(dims_, dims_);
        return get_input_mat_impl(dt, R);
    }

    /// Input matrix G with rotation.
    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt,
        const Vector& /*state*/,
        const RotationMatrix& rotation) const override {

        return get_input_mat_impl(dt, rotation);
    }

private:
    int dims_;

    [[nodiscard]] Matrix get_state_mat_impl(
        Scalar dt, const RotationMatrix& rotation) const {

        const int n = dims_;
        // F = [I, R*dt; 0, I]
        Matrix F = Matrix::Identity(2 * n, 2 * n);
        F.block(0, n, n, n) = dt * rotation.topLeftCorner(n, n);
        return F;
    }

    [[nodiscard]] InputMatrix get_input_mat_impl(
        Scalar dt, const RotationMatrix& rotation) const {

        const int n = dims_;
        // G = [R*dt; I]  — process noise enters in body frame
        InputMatrix G = InputMatrix::Zero(2 * n, n);
        G.block(0, 0, n, n) = dt * rotation.topLeftCorner(n, n);
        G.block(n, 0, n, n) =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        return G;
    }
};

} // namespace brew::dynamics
