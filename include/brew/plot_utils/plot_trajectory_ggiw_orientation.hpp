#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/models/trajectory_ggiw_orientation.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot 2D TrajectoryGGIWOrientation: trajectory line + GGIW extent ellipses
/// + principal axis lines at last state.
void plot_trajectory_ggiw_orientation_2d(matplot::axes_handle ax,
                                         const brew::models::TrajectoryGGIWOrientation& tg,
                                         const std::vector<int>& plt_inds,
                                         const Color& color,
                                         double confidence = 0.99);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_trajectory_ggiw_orientation(matplot::axes_handle ax,
                                      const brew::models::TrajectoryGGIWOrientation& tg,
                                      const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_trajectory_ggiw_orientation(const brew::models::TrajectoryGGIWOrientation& tg,
                                      const PlotOptions& opts);

} // namespace brew::plot_utils
