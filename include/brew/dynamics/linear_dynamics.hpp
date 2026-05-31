#pragma once

#include "brew/dynamics/dynamics_base.hpp"

namespace brew::dynamics {

/// Linear dynamics model: propagate_state = F*x + G*u.
/// Subclasses only need to define get_state_mat() and get_input_mat().
template <typename Scalar = double, int D = Eigen::Dynamic>
class LinearDynamics : public DynamicsBase<Scalar, D> {
public:
    using Base = DynamicsBase<Scalar, D>;
    using Vector = typename Base::Vector;
    using InputVector = typename Base::InputVector;

    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& input = InputVector{}) const override {

        const auto F = this->get_state_mat(dt, state);
        const auto G = this->get_input_mat(dt, state);
        const InputVector u = input.size() > 0
            ? input : InputVector::Zero(G.cols());
        return F * state + G * u;
    }
};

} // namespace brew::dynamics
