#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace brew::dynamics {

/// Abstract base class for dynamics models.
class DynamicsBase {
public:
    virtual ~DynamicsBase() = default;

    [[nodiscard]] virtual std::unique_ptr<DynamicsBase> clone() const = 0;

    /// State names (compile-time constant per concrete class).
    [[nodiscard]] virtual std::vector<std::string> state_names() const = 0;

    /// Propagate state forward by dt.
    [[nodiscard]] virtual Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const = 0;

    /// Propagate state forward by dt with orientation coupling.
    /// Override for body-frame dynamics where velocity depends on rotation.
    [[nodiscard]] virtual Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& rotation,
        const Eigen::VectorXd& input = {}) const {
        return propagate_state(dt, state, input);
    }

    /// Get the state transition matrix F (possibly state-dependent).
    [[nodiscard]] virtual Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const = 0;

    /// Get the state transition matrix F with orientation coupling.
    [[nodiscard]] virtual Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& rotation) const {
        return get_state_mat(dt, state);
    }

    /// Get the input matrix G (possibly state-dependent).
    [[nodiscard]] virtual Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& state = {}) const = 0;

    /// Get the input matrix G with orientation coupling.
    [[nodiscard]] virtual Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& rotation) const {
        return get_input_mat(dt, state);
    }

    /// Propagate an extent matrix forward by dt. Default is identity (no rotation).
    /// Override in models where rotation is part of the state (e.g. ConstantTurn2D).
    [[nodiscard]] virtual Eigen::MatrixXd propagate_extent(
        double /*dt*/,
        const Eigen::VectorXd& /*state*/,
        const Eigen::MatrixXd& extent) const {
        return extent;
    }
};

} // namespace brew::dynamics
