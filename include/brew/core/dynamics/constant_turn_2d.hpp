#pragma once

#include "dynamics_base.hpp"
#include <cmath>

namespace brew::dynamics {

/// 2D constant turn-rate model: state = [x, y, v, theta, omega].
// @mex dynamics
// @mex_name ConstantTurn2D
template <typename Scalar = double>
class ConstantTurn2D : public DynamicsBase<Scalar, 5> {
public:
    using Base = DynamicsBase<Scalar, 5>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;
    using ExtentMatrix = typename Base::ExtentMatrix;

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, 5>> clone() const override {
        return std::make_unique<ConstantTurn2D<Scalar>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "v", "theta", "omega"};
    }

    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& input = InputVector{}) const override {

        InputVector u = input.size() > 0 ? input : InputVector::Zero(2);
        return get_state_mat(dt, state) * state + get_input_mat(dt, state) * u;
    }

    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& state = Vector{}) const override {

        Scalar v = state(2), theta = state(3);

        Matrix F = Matrix::Identity();
        F(0, 2) = std::cos(theta) * dt;
        F(0, 3) = -v * std::sin(theta) * dt;
        F(1, 2) = std::sin(theta) * dt;
        F(1, 3) = v * std::cos(theta) * dt;
        F(3, 4) = dt;
        return F;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt,
        const Vector& state = Vector{}) const override {

        const Scalar theta = state(3);
        InputMatrix G = InputMatrix::Zero(5, 2);
        G(0, 0) = dt * std::cos(theta);
        G(1, 0) = dt * std::sin(theta);
        G(2, 0) = Scalar(1);
        G(4, 1) = Scalar(1);
        return G;
    }

    [[nodiscard]] ExtentMatrix propagate_extent(
        Scalar dt,
        const Vector& state,
        const ExtentMatrix& extent) const override {
        const Scalar omega = state(4);
        const Scalar dtheta = omega * dt;
        const Scalar c = std::cos(dtheta);
        const Scalar s = std::sin(dtheta);
        Eigen::Matrix<Scalar, 2, 2> R;
        R << c, -s,
             s,  c;
        return R * extent * R.transpose();
    }
};

} // namespace brew::dynamics
