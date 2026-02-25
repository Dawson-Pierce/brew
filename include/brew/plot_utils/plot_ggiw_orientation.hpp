#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/models/ggiw_orientation.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot a 2D GGIWOrientation: mean extent ellipse + confidence + principal axis lines.
/// The principal axes are drawn through the center along each eigenvector,
/// scaled by sqrt(eigenvalue).
void plot_ggiw_orientation_2d(matplot::axes_handle ax,
                              const brew::models::GGIWOrientation& g,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence = 0.99);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_ggiw_orientation(matplot::axes_handle ax,
                           const brew::models::GGIWOrientation& g,
                           const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_ggiw_orientation(const brew::models::GGIWOrientation& g,
                           const PlotOptions& opts);

} // namespace brew::plot_utils
