#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include <matplot/matplot.h>

namespace brew::plot_utils {

void plot_point_cloud_2d(matplot::axes_handle ax,
                         const brew::template_matching::PointCloud& cloud,
                         const std::vector<int>& plt_inds,
                         const Color& color,
                         float marker_size = 6.0f);

void plot_point_cloud_3d(matplot::axes_handle ax,
                         const brew::template_matching::PointCloud& cloud,
                         const std::vector<int>& plt_inds,
                         const Color& color,
                         float marker_size = 6.0f);

void plot_point_cloud(matplot::axes_handle ax,
                      const brew::template_matching::PointCloud& cloud,
                      const PlotOptions& opts);

void plot_point_cloud(const brew::template_matching::PointCloud& cloud,
                      const PlotOptions& opts);

}
