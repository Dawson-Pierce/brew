#pragma once

// Ported from: +BREW/+dynamics/DynamicsBase.m
// Original name: DynamicsBase
// Ported on: 2026-02-07
// Notes: Provides rotation model hook for extent propagation.

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace brew::dynamics {

/// Abstract base class for dynamics models.
/// Mirrors MATLAB: BREW.dynamics.DynamicsBase
class DynamicsBase {
public:
    /// Rotation model: either a constant matrix or a state-dependent function.
    using RotationFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& state, double dt)>;
    using RotationModel = std::variant<std::monostate, Eigen::MatrixXd, RotationFunc>;

    virtual ~DynamicsBase() = default;

    [[nodiscard]] virtual std::unique_ptr<DynamicsBase> clone() const = 0;

    /// State names (compile-time constant per concrete class).
    [[nodiscard]] virtual std::vector<std::string> state_names() const = 0;

    /// Propagate state forward by dt.
    [[nodiscard]] virtual Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const = 0;

    /// Get the state transition matrix F (possibly state-dependent).
    [[nodiscard]] virtual Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const = 0;

    /// Get the input matrix G (possibly state-dependent).
    [[nodiscard]] virtual Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const = 0;

    /// Set a rotation model for extent propagation.
    void set_rotation_model(RotationModel M) { M_ = std::move(M); }

    /// Propagate the extent matrix through the rotation model.
    [[nodiscard]] virtual Eigen::MatrixXd propagate_extent(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& extent) const {

        if (std::holds_alternative<std::monostate>(M_)) {
            return extent;
        }
        Eigen::MatrixXd R;
        if (auto* mat = std::get_if<Eigen::MatrixXd>(&M_)) {
            R = *mat;
        } else {
            R = std::get<RotationFunc>(M_)(state, dt);
        }
        return R * extent * R.transpose();
    }

protected:
    RotationModel M_;
};

} // namespace brew::dynamics
