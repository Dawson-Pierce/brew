#pragma once

// Umbrella header for the `trajectory_ggiw_orientation` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/trajectory_ggiw_orientation/trajectory_ggiw_orientation_model.hpp"
#include "brew/ggiw_orientation/ggiw_orientation_model.hpp"

#include "brew/trajectory_ggiw_orientation/filters/trajectory_ggiw_orientation_ekf.hpp"

#include "brew/trajectory_ggiw_orientation/multi_target/phd.hpp"
#include "brew/trajectory_ggiw_orientation/multi_target/mbm_base.hpp"
#include "brew/trajectory_ggiw_orientation/multi_target/cphd.hpp"
#include "brew/trajectory_ggiw_orientation/multi_target/glmb.hpp"
#include "brew/trajectory_ggiw_orientation/multi_target/jglmb.hpp"
#include "brew/trajectory_ggiw_orientation/multi_target/mbm.hpp"
#include "brew/trajectory_ggiw_orientation/multi_target/pmbm.hpp"

#include "brew/trajectory_ggiw_orientation/merge.hpp"
#include "brew/trajectory_ggiw_orientation/gci.hpp"
#include "brew/shared/fusion/arithmetic_average.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
