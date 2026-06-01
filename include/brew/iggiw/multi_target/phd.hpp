#pragma once

// PHD multi-object filter for the `iggiw` package. Devirtualized: holds the
// concrete IGGIW filter by value (see default_filter trait).
#include "brew/iggiw/filters/iggiw_ekf.hpp"
#include "brew/shared/multi_target_generic/phd.hpp"

namespace brew::iggiw {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
using PHD = brew::multi_target::PHD<models::IGGIW<Scalar, D, De>, MaxComponents>;

}  // namespace brew::iggiw
