#pragma once

// CPHD multi-object filter for the `trajectory_gaussian` package. Devirtualized: holds the
// concrete TrajectoryGaussian filter by value (see default_filter trait).
#include "brew/trajectory_gaussian/filters/trajectory_gaussian_ekf.hpp"
#include "brew/shared/multi_target_generic/cphd.hpp"

namespace brew::trajectory_gaussian {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic>
using CPHD = brew::multi_target::CPHD<models::TrajectoryGaussian<Scalar, D>, MaxComponents>;

}  // namespace brew::trajectory_gaussian
