#pragma once

// GLMB multi-object filter for the `trajectory_iggiw` package. Devirtualized: holds the
// concrete TrajectoryIGGIW filter by value (see default_filter trait).
#include "brew/trajectory_iggiw/filters/trajectory_iggiw_ekf.hpp"
#include "brew/shared/multi_target_generic/glmb.hpp"

namespace brew::trajectory_iggiw {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
using GLMB = brew::multi_target::GLMB<models::TrajectoryIGGIW<Scalar, D, De>, MaxComponents>;

}  // namespace brew::trajectory_iggiw
