#pragma once

// Umbrella header for the `trajectory_ggiw` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/trajectory_ggiw/trajectory_ggiw_model.hpp"
#include "brew/ggiw/ggiw_model.hpp"

#include "brew/trajectory_ggiw/filters/trajectory_ggiw_ekf.hpp"

#include "brew/trajectory_ggiw/multi_target/phd.hpp"
#include "brew/trajectory_ggiw/multi_target/mbm_base.hpp"
#include "brew/trajectory_ggiw/multi_target/cphd.hpp"
#include "brew/trajectory_ggiw/multi_target/glmb.hpp"
#include "brew/trajectory_ggiw/multi_target/jglmb.hpp"
#include "brew/trajectory_ggiw/multi_target/mbm.hpp"
#include "brew/trajectory_ggiw/multi_target/pmbm.hpp"

#include "brew/trajectory_ggiw/merge.hpp"
#include "brew/trajectory_ggiw/gci.hpp"
#include "brew/shared/fusion/arithmetic_average.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
