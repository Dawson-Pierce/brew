#pragma once

// PHD multi-object filter for the `gaussian` package. Devirtualized: holds the
// concrete Gaussian filter by value (see default_filter trait).
#include "brew/gaussian/filters/ekf.hpp"
#include "brew/shared/multi_target_generic/phd.hpp"

namespace brew::gaussian {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic>
using PHD = brew::multi_target::PHD<models::Gaussian<Scalar, D>, MaxComponents>;

}  // namespace brew::gaussian
