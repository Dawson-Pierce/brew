#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/models/ggiw.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot a 2D GGIW: mean extent ellipse (solid) + confidence interval ellipse (dashed).
/// plt_inds must contain exactly 2 indices.
void plot_ggiw_2d(matplot::axes_handle ax,
                  const brew::models::GGIW& g,
                  const std::vector<int>& plt_inds,
                  const Color& color,
                  double confidence = 0.99);

/// Plot a 3D GGIW: mean extent ellipsoid + confidence ellipsoid.
/// plt_inds must contain exactly 3 indices.
void plot_ggiw_3d(matplot::axes_handle ax,
                  const brew::models::GGIW& g,
                  const std::vector<int>& plt_inds,
                  const Color& color,
                  double confidence = 0.99,
                  float alpha = 0.3f,
                  float colormap_value = 0.0f);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_ggiw(matplot::axes_handle ax,
               const brew::models::GGIW& g,
               const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_ggiw(const brew::models::GGIW& g,
               const PlotOptions& opts);

} // namespace brew::plot_utils

