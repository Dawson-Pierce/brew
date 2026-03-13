#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot a 2D point cloud scatter on the given axes.
/// plt_inds must contain exactly 2 indices (row indices into the points matrix).
void plot_point_cloud_2d(matplot::axes_handle ax,
                         const brew::template_matching::PointCloud& cloud,
                         const std::vector<int>& plt_inds,
                         const Color& color,
                         float marker_size = 6.0f);

/// Plot a 3D point cloud scatter on the given axes.
/// plt_inds must contain exactly 3 indices (row indices into the points matrix).
void plot_point_cloud_3d(matplot::axes_handle ax,
                         const brew::template_matching::PointCloud& cloud,
                         const std::vector<int>& plt_inds,
                         const Color& color,
                         float marker_size = 6.0f);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_point_cloud(matplot::axes_handle ax,
                      const brew::template_matching::PointCloud& cloud,
                      const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_point_cloud(const brew::template_matching::PointCloud& cloud,
                      const PlotOptions& opts);

} // namespace brew::plot_utils
