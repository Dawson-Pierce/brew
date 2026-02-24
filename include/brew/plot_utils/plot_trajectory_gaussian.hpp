#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/models/trajectory_gaussian.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot 1D TrajectoryGaussian: state value vs time index + window overlay.
/// plt_inds must contain exactly 1 index.
void plot_trajectory_gaussian_1d(matplot::axes_handle ax,
                                  const brew::models::TrajectoryGaussian& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color);

/// Plot 2D TrajectoryGaussian: trajectory line from rearrange_states + window overlay.
/// plt_inds must contain exactly 2 indices.
void plot_trajectory_gaussian_2d(matplot::axes_handle ax,
                                  const brew::models::TrajectoryGaussian& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color);

/// Plot 3D TrajectoryGaussian: plot3 trajectory + window.
/// plt_inds must contain exactly 3 indices.
void plot_trajectory_gaussian_3d(matplot::axes_handle ax,
                                  const brew::models::TrajectoryGaussian& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_trajectory_gaussian(matplot::axes_handle ax,
                               const brew::models::TrajectoryGaussian& tg,
                               const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_trajectory_gaussian(const brew::models::TrajectoryGaussian& tg,
                               const PlotOptions& opts);

} // namespace brew::plot_utils

