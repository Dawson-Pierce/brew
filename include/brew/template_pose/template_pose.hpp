#pragma once

// Umbrella header for the `template_pose` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/template_pose/template_pose_model.hpp"

#include "brew/template_pose/filters/tm_ekf.hpp"

#include "brew/template_pose/multi_target/phd.hpp"
#include "brew/template_pose/multi_target/mbm_base.hpp"
#include "brew/template_pose/multi_target/cphd.hpp"
#include "brew/template_pose/multi_target/glmb.hpp"
#include "brew/template_pose/multi_target/jglmb.hpp"
#include "brew/template_pose/multi_target/mbm.hpp"
#include "brew/template_pose/multi_target/pmbm.hpp"

#include "brew/template_pose/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
