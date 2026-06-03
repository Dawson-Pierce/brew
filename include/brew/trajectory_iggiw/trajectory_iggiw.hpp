#pragma once

// Umbrella header for the `trajectory_iggiw` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/trajectory_iggiw/trajectory_iggiw_model.hpp"
#include "brew/iggiw/iggiw_model.hpp"

#include "brew/trajectory_iggiw/filters/trajectory_iggiw_ekf.hpp"

#include "brew/trajectory_iggiw/multi_target/phd.hpp"
#include "brew/trajectory_iggiw/multi_target/mbm_base.hpp"
#include "brew/trajectory_iggiw/multi_target/cphd.hpp"
#include "brew/trajectory_iggiw/multi_target/glmb.hpp"
#include "brew/trajectory_iggiw/multi_target/jglmb.hpp"
#include "brew/trajectory_iggiw/multi_target/mbm.hpp"
#include "brew/trajectory_iggiw/multi_target/pmbm.hpp"

#include "brew/trajectory_iggiw/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
