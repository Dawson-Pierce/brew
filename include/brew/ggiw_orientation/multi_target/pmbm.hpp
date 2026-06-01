#pragma once

// PMBM multi-object filter for the `ggiw_orientation` package. Devirtualized: holds the
// concrete GGIWOrientation filter by value (see default_filter trait).
#include "brew/ggiw_orientation/filters/ggiw_orientation_ekf.hpp"
#include "brew/shared/multi_target_generic/pmbm.hpp"

namespace brew::ggiw_orientation {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
using PMBM = brew::multi_target::PMBM<models::GGIWOrientation<Scalar, D, De>, MaxComponents>;

}  // namespace brew::ggiw_orientation
