#pragma once

// JGLMB multi-object filter for the `trajectory_ggiw` package. Devirtualized: holds the
// concrete TrajectoryGGIW filter by value (see default_filter trait).
#include "brew/trajectory_ggiw/filters/trajectory_ggiw_ekf.hpp"
#include "brew/shared/multi_target_generic/jglmb.hpp"

namespace brew::trajectory_ggiw {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
using JGLMB = brew::multi_target::JGLMB<models::TrajectoryGGIW<Scalar, D, De>, MaxComponents>;

}  // namespace brew::trajectory_ggiw
