#include "brew/dynamics/linear_dynamics.hpp"

namespace brew::dynamics {

Eigen::VectorXd LinearDynamics::propagate_state(
    double dt,
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& input) const {

    Eigen::VectorXd u = input.size() > 0
        ? input
        : Eigen::VectorXd::Zero(get_input_mat(dt, state).cols());
    return get_state_mat(dt, state) * state + get_input_mat(dt, state) * u;
}

} // namespace brew::dynamics
