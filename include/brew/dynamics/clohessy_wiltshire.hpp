#pragma once

#include <cmath>

#include "brew/assert.hpp"
#include "brew/dynamics/linear_dynamics.hpp"

namespace brew::dynamics {

// Exact Clohessy-Wiltshire relative-motion transition about a circular orbit.
// State ordering is [x, y, z, vx, vy, vz] in the Hill frame.
// @mex dynamics
// @mex_name ClohessyWiltshire
// @mex_args mean_motion:double
template <typename Scalar = double>
class ClohessyWiltshire : public LinearDynamics<Scalar, Eigen::Dynamic> {
public:
    using Base = LinearDynamics<Scalar, Eigen::Dynamic>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputMatrix = typename Base::InputMatrix;

    explicit ClohessyWiltshire(Scalar mean_motion)
        : mean_motion_(mean_motion) {
        BREW_ASSERT(mean_motion > Scalar(0),
            "ClohessyWiltshire: mean motion must be positive");
    }

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, Eigen::Dynamic>> clone() const override {
        return std::make_unique<ClohessyWiltshire<Scalar>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "z", "vx", "vy", "vz"};
    }

    [[nodiscard]] Matrix get_state_mat(
        Scalar dt, const Vector& = Vector{}) const override {
        const Scalar n = mean_motion_;
        const Scalar nt = n * dt;
        const Scalar c = std::cos(nt);
        const Scalar s = std::sin(nt);
        Matrix F = Matrix::Zero(6, 6);
        F(0, 0) = Scalar(4) - Scalar(3) * c;
        F(0, 3) = s / n;
        F(0, 4) = Scalar(2) * (Scalar(1) - c) / n;
        F(1, 0) = Scalar(6) * (s - nt);
        F(1, 1) = Scalar(1);
        F(1, 3) = -Scalar(2) * (Scalar(1) - c) / n;
        F(1, 4) = (Scalar(4) * s - Scalar(3) * nt) / n;
        F(2, 2) = c;
        F(2, 5) = s / n;
        F(3, 0) = Scalar(3) * n * s;
        F(3, 3) = c;
        F(3, 4) = Scalar(2) * s;
        F(4, 0) = -Scalar(6) * n * (Scalar(1) - c);
        F(4, 3) = -Scalar(2) * s;
        F(4, 4) = Scalar(4) * c - Scalar(3);
        F(5, 2) = -n * s;
        F(5, 5) = c;
        return F;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt, const Vector& = Vector{}) const override {
        const Scalar n = mean_motion_;
        const Scalar nt = n * dt;
        const Scalar c = std::cos(nt);
        const Scalar s = std::sin(nt);
        const Scalar n2 = n * n;
        InputMatrix G = InputMatrix::Zero(6, 3);
        G(0, 0) = (Scalar(1) - c) / n2;
        G(0, 1) = Scalar(2) * (nt - s) / n2;
        G(1, 0) = Scalar(2) * (s - nt) / n2;
        G(1, 1) = Scalar(4) * (Scalar(1) - c) / n2
                  - Scalar(1.5) * dt * dt;
        G(2, 2) = (Scalar(1) - c) / n2;
        G(3, 0) = s / n;
        G(3, 1) = Scalar(2) * (Scalar(1) - c) / n;
        G(4, 0) = -Scalar(2) * (Scalar(1) - c) / n;
        G(4, 1) = Scalar(4) * s / n - Scalar(3) * dt;
        G(5, 2) = s / n;
        return G;
    }

    [[nodiscard]] bool is_lti() const override { return true; }
    [[nodiscard]] Scalar mean_motion() const { return mean_motion_; }

    [[nodiscard]] Eigen::Matrix<Scalar, 3, 1> acceleration(const Vector& state) const {
        BREW_ASSERT(state.size() == 6,
            "ClohessyWiltshire: state must contain six elements");
        const Scalar n = mean_motion_;
        Eigen::Matrix<Scalar, 3, 1> value;
        value << Scalar(3) * n * n * state(0) + Scalar(2) * n * state(4),
                 -Scalar(2) * n * state(3),
                 -n * n * state(2);
        return value;
    }

private:
    Scalar mean_motion_;
};

}
