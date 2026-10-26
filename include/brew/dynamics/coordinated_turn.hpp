#pragma once

#include "dynamics_base.hpp"
#include <cmath>

namespace brew::dynamics {

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

    [[nodiscard]] Matrix get_state_mat(
        Scalar dt, const Vector& state = Vector{}) const override {

        Matrix F = Matrix::Identity(kDim, kDim);
        const Scalar w = (state.size() >= kDim) ? state(4) : Scalar(0);
        if (std::abs(w) < Scalar(1e-6)) {

            F(0, 2) = dt;
            F(1, 3) = dt;
            F(0, 4) = -Scalar(0.5) * dt * dt * state(3);
            F(1, 4) =  Scalar(0.5) * dt * dt * state(2);
            F(2, 4) = -dt * state(3);
            F(3, 4) =  dt * state(2);
        } else {
            const Scalar s = std::sin(w * dt);
            const Scalar c = std::cos(w * dt);
            const Scalar a = s / w;
            const Scalar b = (Scalar(1) - c) / w;
            const Scalar da = (w * dt * c - s) / (w * w);
            const Scalar db = (w * dt * s - (Scalar(1) - c)) / (w * w);
            F(0, 2) = a; F(0, 3) = -b;
            F(1, 2) = b; F(1, 3) = a;
            F(2, 2) = c;                   F(2, 3) = -s;
            F(3, 2) = s;                   F(3, 3) = c;
            F(0, 4) = da * state(2) - db * state(3);
            F(1, 4) = db * state(2) + da * state(3);
            F(2, 4) = -dt * s * state(2) - dt * c * state(3);
            F(3, 4) =  dt * c * state(2) - dt * s * state(3);
        }
        return F;
    }

    [[nodiscard]] Vector propagate_state(
        Scalar dt, const Vector& state,
        const InputVector&  = InputVector{}) const override {
        Vector next = state;
        const Scalar w = state(4);
        const Scalar vx = state(2);
        const Scalar vy = state(3);
        if (std::abs(w) < Scalar(1e-6)) {
            next(0) += dt * vx;
            next(1) += dt * vy;
        } else {
            const Scalar s = std::sin(w * dt);
            const Scalar c = std::cos(w * dt);
            const Scalar a = s / w;
            const Scalar b = (Scalar(1) - c) / w;
            next(0) += a * vx - b * vy;
            next(1) += b * vx + a * vy;
            next(2) = c * vx - s * vy;
            next(3) = s * vx + c * vy;
        }
        return next;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar , const Vector& = Vector{}) const override {
        return InputMatrix::Zero(kDim, 1);
    }
};

}
