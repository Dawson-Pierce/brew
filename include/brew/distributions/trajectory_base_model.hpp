#pragma once

// Ported from: +BREW/+distributions/TrajectoryBaseModel.m
// Original name: TrajectoryBaseModel
// Ported on: 2026-02-07
// Notes: Header-only base with metadata.

#include <cstddef>

namespace brew::distributions {

/// Base model for trajectory distributions (alternative to labeled RFS).
/// Mirrors MATLAB: BREW.distributions.TrajectoryBaseModel
struct TrajectoryBaseModel {
    int state_dim = 0;      ///< Length of the state vector
    int window_size = 1;    ///< Length of the trajectory
    int init_idx = 0;       ///< Time index when trajectory was created

    TrajectoryBaseModel() = default;
    TrajectoryBaseModel(int idx, int state_dim)
        : state_dim(state_dim), window_size(1), init_idx(idx) {}

    virtual ~TrajectoryBaseModel() = default;
};

} // namespace brew::distributions
