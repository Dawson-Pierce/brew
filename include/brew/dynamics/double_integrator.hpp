#pragma once

#include "linear_dynamics.hpp"
#include <stdexcept>

namespace brew::dynamics {

/// Constant-acceleration (double) integrator for 1, 2, or 3 spatial dimensions.
/// State ordering: [pos..., vel..., acc...].
/// Input is an acceleration increment applied each step.
class DoubleIntegrator : public LinearDynamics {
public:
    explicit DoubleIntegrator(int dims) : dims_(dims) {
        if (dims < 1 || dims > 3)
            throw std::invalid_argument("DoubleIntegrator: dims must be 1, 2, or 3");
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<DoubleIntegrator>(*this);
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
    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& = {}) const override {

        const int n = dims_;
        const auto In = Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(3*n, 3*n);
        F.block(  0,   n, n, n) = dt * In;
        F.block(  0, 2*n, n, n) = 0.5 * dt * dt * In;
        F.block(  n, 2*n, n, n) = dt * In;
        return F;
    }

    /// G = kron([[0.5*dt^2], [dt], [1]], I_dims)
    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& = {}) const override {

        const int n = dims_;
        const auto In = Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(3*n, n);
        G.block(  0, 0, n, n) = 0.5 * dt * dt * In;
        G.block(  n, 0, n, n) = dt * In;
        G.block(2*n, 0, n, n) = In;
        return G;
    }

private:
    int dims_;
};

} // namespace brew::dynamics
