#pragma once

// Umbrella header for the `gaussian` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/gaussian/gaussian_model.hpp"

#include "brew/gaussian/filters/ekf.hpp"

#include "brew/gaussian/multi_target/phd.hpp"
#include "brew/gaussian/multi_target/mbm_base.hpp"
#include "brew/gaussian/multi_target/cphd.hpp"
#include "brew/gaussian/multi_target/glmb.hpp"
#include "brew/gaussian/multi_target/jglmb.hpp"
#include "brew/gaussian/multi_target/mbm.hpp"
#include "brew/gaussian/multi_target/pmbm.hpp"

#include "brew/gaussian/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
