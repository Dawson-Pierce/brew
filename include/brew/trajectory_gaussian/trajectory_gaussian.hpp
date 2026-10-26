#pragma once

// Umbrella header for the `trajectory_gaussian` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/trajectory_gaussian/trajectory_gaussian_model.hpp"
#include "brew/gaussian/gaussian_model.hpp"

#include "brew/trajectory_gaussian/filters/trajectory_gaussian_ekf.hpp"

#include "brew/trajectory_gaussian/multi_target/phd.hpp"
#include "brew/trajectory_gaussian/multi_target/mbm_base.hpp"
#include "brew/trajectory_gaussian/multi_target/cphd.hpp"
#include "brew/trajectory_gaussian/multi_target/glmb.hpp"
#include "brew/trajectory_gaussian/multi_target/jglmb.hpp"
#include "brew/trajectory_gaussian/multi_target/mbm.hpp"
#include "brew/trajectory_gaussian/multi_target/pmbm.hpp"

#include "brew/trajectory_gaussian/merge.hpp"
#include "brew/trajectory_gaussian/gci.hpp"
#include "brew/shared/fusion/arithmetic_average.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
