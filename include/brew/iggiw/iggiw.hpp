#pragma once

// Umbrella header for the `iggiw` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/iggiw/iggiw_model.hpp"

#include "brew/iggiw/filters/iggiw_ekf.hpp"

#include "brew/iggiw/multi_target/phd.hpp"
#include "brew/iggiw/multi_target/mbm_base.hpp"
#include "brew/iggiw/multi_target/cphd.hpp"
#include "brew/iggiw/multi_target/glmb.hpp"
#include "brew/iggiw/multi_target/jglmb.hpp"
#include "brew/iggiw/multi_target/mbm.hpp"
#include "brew/iggiw/multi_target/pmbm.hpp"

#include "brew/iggiw/merge.hpp"
#include "brew/iggiw/gci.hpp"
#include "brew/shared/fusion/arithmetic_average.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
