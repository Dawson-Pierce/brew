#pragma once

#include "dynamics_base.hpp"
#include <cmath>

namespace brew::dynamics {

/// 2D coordinated-turn model with the turn rate as an UNKNOWN, estimated state.
///
///     state = [x, y, vx, vy, omega]
///             (Cartesian position + velocity + turn rate, rad/s)
///
/// The position/velocity transition F(omega, dt) depends on omega — which is part
/// of the state — so the model is genuinely NONLINEAR and not linear-time-invariant
/// (is_lti() stays false). That makes it the natural test case for the UKF, which
/// propagates sigma points through the exact transition, versus the EKF, which
/// linearizes (and, with the Jacobian below, does not couple omega to the measured
/// position — so it cannot recover an unknown turn rate from position alone).
///
/// Given omega the map on [x, y, vx, vy] is linear, so propagate_state is the EXACT
/// F(omega, dt) * state (no Euler/first-order approximation); the only nonlinearity
/// is that omega (a state element) parameterizes F. As omega -> 0 it reduces to a
/// constant-velocity model.
///
/// Built on DynamicsBase<Scalar, Eigen::Dynamic> (runtime 5-dim) so it pairs with
/// the standard Gaussian<Scalar, Dynamic> filters and is usable from MATLAB.
// @mex dynamics
// @mex_name CoordinatedTurn
template <typename Scalar = double>
class CoordinatedTurn : public DynamicsBase<Scalar, Eigen::Dynamic> {
public:
    using Base = DynamicsBase<Scalar, Eigen::Dynamic>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;

    static constexpr int kDim = 5;

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, Eigen::Dynamic>> clone() const override {
        return std::make_unique<CoordinatedTurn<Scalar>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "vx", "vy", "omega"};
    }

    /// Exact coordinated-turn transition for the omega carried in `state` (index 4).
    [[nodiscard]] Matrix get_state_mat(
        Scalar dt, const Vector& state = Vector{}) const override {

        Matrix F = Matrix::Identity(kDim, kDim);
        const Scalar w = (state.size() >= kDim) ? state(4) : Scalar(0);
        if (std::abs(w) < Scalar(1e-6)) {
            // Constant-velocity limit (omega ~ 0).
            F(0, 2) = dt;
            F(1, 3) = dt;
        } else {
            const Scalar s = std::sin(w * dt);
            const Scalar c = std::cos(w * dt);
            F(0, 2) = s / w;               F(0, 3) = -(Scalar(1) - c) / w;
            F(1, 2) = (Scalar(1) - c) / w; F(1, 3) = s / w;
            F(2, 2) = c;                   F(2, 3) = -s;
            F(3, 2) = s;                   F(3, 3) = c;
        }
        return F;  // row/col 4 (omega) stays identity: omega is constant over a step
    }

    /// Exact propagation: F(omega, dt) * state (linear in [x,y,vx,vy] given omega).
    [[nodiscard]] Vector propagate_state(
        Scalar dt, const Vector& state,
        const InputVector& /*input*/ = InputVector{}) const override {
        return get_state_mat(dt, state) * state;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar /*dt*/, const Vector& = Vector{}) const override {
        return InputMatrix::Zero(kDim, 1);  // no control input modeled
    }
};

} // namespace brew::dynamics
