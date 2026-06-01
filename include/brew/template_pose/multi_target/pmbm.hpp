#pragma once

// PMBM multi-object filter for the `template_pose` package. Devirtualized: holds the
// concrete TemplatePose filter by value (see default_filter trait).
#include "brew/template_pose/filters/tm_ekf.hpp"
#include "brew/shared/multi_target_generic/pmbm.hpp"

namespace brew::template_pose {

template <int MaxComponents = Eigen::Dynamic, typename Scalar = double, int D = Eigen::Dynamic>
using PMBM = brew::multi_target::PMBM<models::TemplatePose<Scalar, D>, MaxComponents>;

}  // namespace brew::template_pose
