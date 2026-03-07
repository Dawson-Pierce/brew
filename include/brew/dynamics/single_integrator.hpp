#pragma once

#include "linear_dynamics.hpp"
#include <stdexcept>

namespace brew::dynamics {

/// Constant-velocity integrator for 1, 2, or 3 spatial dimensions.
/// State ordering: [pos..., vel...].
/// Input is a velocity increment applied each step.
class SingleIntegrator : public LinearDynamics {
public:
    explicit SingleIntegrator(int dims) : dims_(dims) {
        if (dims < 1 || dims > 3)
            throw std::invalid_argument("SingleIntegrator: dims must be 1, 2, or 3");
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase> clone() const override {
        return std::make_unique<SingleIntegrator>(*this);
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
    [[nodiscard]] Eigen::MatrixXd get_state_mat(
        double dt,
        const Eigen::VectorXd& = {}) const override {

        const int n = dims_;
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2*n, 2*n);
        F.block(0, n, n, n) = dt * Eigen::MatrixXd::Identity(n, n);
        return F;
    }

    /// G = kron([[dt], [1]], I_dims)
    [[nodiscard]] Eigen::MatrixXd get_input_mat(
        double dt,
        const Eigen::VectorXd& = {}) const override {

        const int n = dims_;
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(2*n, n);
        G.block(0, 0, n, n) = dt * Eigen::MatrixXd::Identity(n, n);
        G.block(n, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
        return G;
    }

private:
    int dims_;
};

} // namespace brew::dynamics
