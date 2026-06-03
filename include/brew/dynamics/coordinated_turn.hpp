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
        } else {
            const Scalar s = std::sin(w * dt);
            const Scalar c = std::cos(w * dt);
            F(0, 2) = s / w;               F(0, 3) = -(Scalar(1) - c) / w;
            F(1, 2) = (Scalar(1) - c) / w; F(1, 3) = s / w;
            F(2, 2) = c;                   F(2, 3) = -s;
            F(3, 2) = s;                   F(3, 3) = c;
        }
        return F;
    }

    [[nodiscard]] Vector propagate_state(
        Scalar dt, const Vector& state,
        const InputVector&  = InputVector{}) const override {
        return get_state_mat(dt, state) * state;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar , const Vector& = Vector{}) const override {
        return InputMatrix::Zero(kDim, 1);
    }
};

}
