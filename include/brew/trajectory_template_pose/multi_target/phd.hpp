#pragma once

// PHD multi-object filter for the `trajectory_template_pose` package. Devirtualized: holds the
// concrete TrajectoryTemplatePose filter by value (see default_filter trait).
#include "brew/trajectory_template_pose/filters/trajectory_tm_ekf.hpp"
#include "brew/shared/multi_target_generic/phd.hpp"

namespace brew::trajectory_template_pose {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic>
using PHD = brew::multi_target::PHD<models::TrajectoryTemplatePose<Scalar, D>, MaxComponents>;

}  // namespace brew::trajectory_template_pose
