#pragma once

// Umbrella header for the `trajectory_ggiw_orientation` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model
#include "brew/trajectory_ggiw_orientation/trajectory_ggiw_orientation_model.hpp"
#include "brew/ggiw_orientation/ggiw_orientation_model.hpp"

// Single-object filters
#include "brew/trajectory_ggiw_orientation/filters/trajectory_ggiw_orientation_ekf.hpp"

// Multi-object (RFS) filters
#include "brew/shared/multi_target_generic/phd.hpp"
#include "brew/shared/multi_target_generic/cphd.hpp"
#include "brew/shared/multi_target_generic/glmb.hpp"
#include "brew/shared/multi_target_generic/jglmb.hpp"
#include "brew/shared/multi_target_generic/mbm.hpp"
#include "brew/shared/multi_target_generic/pmbm.hpp"

// Mixture management + clustering helpers
#include "brew/shared/fusion/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
