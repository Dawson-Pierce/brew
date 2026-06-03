#pragma once

// Umbrella header for the `ggiw_orientation` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/ggiw_orientation/ggiw_orientation_model.hpp"

#include "brew/ggiw_orientation/filters/ggiw_orientation_ekf.hpp"

#include "brew/ggiw_orientation/multi_target/phd.hpp"
#include "brew/ggiw_orientation/multi_target/mbm_base.hpp"
#include "brew/ggiw_orientation/multi_target/cphd.hpp"
#include "brew/ggiw_orientation/multi_target/glmb.hpp"
#include "brew/ggiw_orientation/multi_target/jglmb.hpp"
#include "brew/ggiw_orientation/multi_target/mbm.hpp"
#include "brew/ggiw_orientation/multi_target/pmbm.hpp"

#include "brew/ggiw_orientation/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
