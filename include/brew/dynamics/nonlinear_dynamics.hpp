#pragma once

// Ported from: +BREW/+dynamics/DynamicsBase.m
// Original name: NonlinearDynamics (C++ helper)
// Ported on: 2026-02-07
// Notes: Marker base for nonlinear dynamics.

#include "brew/dynamics/dynamics_base.hpp"

namespace brew::dynamics {

/// Nonlinear dynamics model.
/// Subclasses must implement propagate_state() with custom equations.
/// get_state_mat() returns the Jacobian for linearization (e.g., EKF).
class NonlinearDynamics : public DynamicsBase {
    // propagate_state remains pure virtual from DynamicsBase
};

} // namespace brew::dynamics
