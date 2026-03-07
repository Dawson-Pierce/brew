#pragma once

#include "brew/dynamics/dynamics_base.hpp"

namespace brew::dynamics {

/// Linear dynamics model: propagate_state = F*x + G*u.
/// Subclasses only need to define get_state_mat() and get_input_mat().
class LinearDynamics : public DynamicsBase {
public:
    /// Propagates state forward: F(dt,state)*state + G(dt,state)*input
    [[nodiscard]] Eigen::VectorXd propagate_state(
        double dt,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& input = {}) const override;
};

} // namespace brew::dynamics
