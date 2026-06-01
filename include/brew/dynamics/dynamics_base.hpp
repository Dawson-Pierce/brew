#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace brew::dynamics {

/// Multiply-or-pass-through for Eigen::Dynamic. Used by derived classes that
/// express state dim as spatial_dim * K but must forward to DynamicsBase<..., D>.
namespace detail {
    constexpr int scaled_dim(int d, int factor) {
        return d == Eigen::Dynamic ? Eigen::Dynamic : d * factor;
    }
}

/// Abstract base class for dynamics models, templated on scalar type and state dimension.
/// D is the full state dimension (e.g. 6 for constant-acceleration 2D tracking).
template <typename Scalar = double, int D = Eigen::Dynamic>
class DynamicsBase {
public:
    using Vector = Eigen::Matrix<Scalar, D, 1>;
    using Matrix = Eigen::Matrix<Scalar, D, D>;
    using InputVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using InputMatrix = Eigen::Matrix<Scalar, D, Eigen::Dynamic>;
    using RotationMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ExtentMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    virtual ~DynamicsBase() = default;

    [[nodiscard]] virtual std::unique_ptr<DynamicsBase> clone() const = 0;

    /// State names (compile-time constant per concrete class).
    [[nodiscard]] virtual std::vector<std::string> state_names() const = 0;

    /// Propagate state forward by dt.
    [[nodiscard]] virtual Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& input = InputVector{}) const = 0;

    /// Propagate state forward by dt with orientation coupling.
    /// Override for body-frame dynamics where velocity depends on rotation.
    [[nodiscard]] virtual Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& /*rotation*/,
        const InputVector& input = InputVector{}) const {
        return propagate_state(dt, state, input);
    }

    /// Get the state transition matrix F (possibly state-dependent).
    [[nodiscard]] virtual Matrix get_state_mat(
        Scalar dt,
        const Vector& state = Vector{}) const = 0;

    /// Get the state transition matrix F with orientation coupling.
    [[nodiscard]] virtual Matrix get_state_mat(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& /*rotation*/) const {
        return get_state_mat(dt, state);
    }

    /// Get the input matrix G (possibly state-dependent).
    [[nodiscard]] virtual InputMatrix get_input_mat(
        Scalar dt,
        const Vector& state = Vector{}) const = 0;

    /// Get the input matrix G with orientation coupling.
    [[nodiscard]] virtual InputMatrix get_input_mat(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& /*rotation*/) const {
        return get_input_mat(dt, state);
    }

    /// Propagate an extent matrix forward by dt. Default is identity (no rotation).
    /// Override in models where rotation is part of the state (e.g. ConstantTurn2D).
    [[nodiscard]] virtual ExtentMatrix propagate_extent(
        Scalar /*dt*/,
        const Vector& /*state*/,
        const ExtentMatrix& extent) const {
        return extent;
    }

    /// True iff the dynamics are linear time-invariant: get_state_mat(dt, state)
    /// is independent of `state` and propagate_state(dt, x) == get_state_mat(dt)*x.
    /// When true, the state-transition F can be built once per dt and shared
    /// across all targets in a batch predict (see Filter::predict_batch). Default
    /// is the conservative false; LTI models (SingleIntegrator/DoubleIntegrator/
    /// Singer) override to true.
    [[nodiscard]] virtual bool is_lti() const { return false; }
};

} // namespace brew::dynamics
