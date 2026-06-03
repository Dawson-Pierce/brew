#pragma once

// Umbrella header for the `trajectory_template_pose` package: model data structure, its
// single-object filters, and the multi-object (RFS) filters usable with it.

// Model

#include "brew/trajectory_template_pose/trajectory_template_pose_model.hpp"
#include "brew/template_pose/template_pose_model.hpp"

#include "brew/trajectory_template_pose/filters/trajectory_tm_ekf.hpp"

#include "brew/trajectory_template_pose/multi_target/phd.hpp"
#include "brew/trajectory_template_pose/multi_target/mbm_base.hpp"
#include "brew/trajectory_template_pose/multi_target/cphd.hpp"
#include "brew/trajectory_template_pose/multi_target/glmb.hpp"
#include "brew/trajectory_template_pose/multi_target/jglmb.hpp"
#include "brew/trajectory_template_pose/multi_target/mbm.hpp"
#include "brew/trajectory_template_pose/multi_target/pmbm.hpp"

#include "brew/trajectory_template_pose/merge.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"
