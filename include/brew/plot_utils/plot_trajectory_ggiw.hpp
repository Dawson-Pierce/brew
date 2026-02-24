#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/models/trajectory_ggiw.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot 2D TrajectoryGGIW: trajectory line + GGIW extent ellipses at last state.
/// plt_inds must contain exactly 2 indices.
void plot_trajectory_ggiw_2d(matplot::axes_handle ax,
                              const brew::models::TrajectoryGGIW& tg,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence = 0.99);

/// Plot 3D TrajectoryGGIW: plot3 trajectory + GGIW extent ellipsoids at last state.
/// plt_inds must contain exactly 3 indices.
void plot_trajectory_ggiw_3d(matplot::axes_handle ax,
                              const brew::models::TrajectoryGGIW& tg,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence = 0.99,
                              float alpha = 0.3f);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_trajectory_ggiw(matplot::axes_handle ax,
                           const brew::models::TrajectoryGGIW& tg,
                           const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_trajectory_ggiw(const brew::models::TrajectoryGGIW& tg,
                           const PlotOptions& opts);

} // namespace brew::plot_utils

