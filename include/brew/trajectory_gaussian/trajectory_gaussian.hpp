#pragma once

// Umbrella header for the `trajectory_gaussian` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model
#include "brew/trajectory_gaussian/trajectory_gaussian_model.hpp"
#include "brew/gaussian/gaussian_model.hpp"

// Single-object filters
#include "brew/trajectory_gaussian/filters/trajectory_gaussian_ekf.hpp"

// Multi-object (RFS) filters
#include "brew/trajectory_gaussian/multi_target/phd.hpp"
#include "brew/trajectory_gaussian/multi_target/mbm_base.hpp"
#include "brew/trajectory_gaussian/multi_target/cphd.hpp"
#include "brew/trajectory_gaussian/multi_target/glmb.hpp"
#include "brew/trajectory_gaussian/multi_target/jglmb.hpp"
#include "brew/trajectory_gaussian/multi_target/mbm.hpp"
#include "brew/trajectory_gaussian/multi_target/pmbm.hpp"

// Mixture management + clustering helpers
#include "brew/trajectory_gaussian/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
