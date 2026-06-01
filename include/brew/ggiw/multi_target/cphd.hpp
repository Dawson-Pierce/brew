#pragma once

// CPHD multi-object filter for the `ggiw` package. Devirtualized: holds the
// concrete GGIW filter by value (see default_filter trait).
#include "brew/ggiw/filters/ggiw_ekf.hpp"
#include "brew/shared/multi_target_generic/cphd.hpp"

namespace brew::ggiw {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
using CPHD = brew::multi_target::CPHD<models::GGIW<Scalar, D, De>, MaxComponents>;

}  // namespace brew::ggiw
