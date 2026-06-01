#pragma once

// Umbrella header for the `trajectory_ggiw` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model
#include "brew/trajectory_ggiw/trajectory_ggiw_model.hpp"
#include "brew/ggiw/ggiw_model.hpp"

// Single-object filters
#include "brew/trajectory_ggiw/filters/trajectory_ggiw_ekf.hpp"

// Multi-object (RFS) filters
#include "brew/trajectory_ggiw/multi_target/phd.hpp"
#include "brew/trajectory_ggiw/multi_target/cphd.hpp"
#include "brew/trajectory_ggiw/multi_target/glmb.hpp"
#include "brew/trajectory_ggiw/multi_target/jglmb.hpp"
#include "brew/trajectory_ggiw/multi_target/mbm.hpp"
#include "brew/trajectory_ggiw/multi_target/pmbm.hpp"

// Mixture management + clustering helpers
#include "brew/shared/fusion/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
