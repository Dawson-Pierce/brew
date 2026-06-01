#pragma once

// JGLMB multi-object filter for the `trajectory_iggiw` package. Devirtualized: holds the
// concrete TrajectoryIGGIW filter by value (see default_filter trait).
#include "brew/trajectory_iggiw/filters/trajectory_iggiw_ekf.hpp"
#include "brew/shared/multi_target_generic/jglmb.hpp"

namespace brew::trajectory_iggiw {

template <int MaxWindow, int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
using JGLMB = brew::multi_target::JGLMB<models::TrajectoryIGGIW<MaxWindow, Scalar, D, De>, MaxComponents>;

}  // namespace brew::trajectory_iggiw
