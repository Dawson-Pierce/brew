#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/distributions/gaussian.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot a 1D Gaussian PDF on the given axes.
/// plt_inds must contain exactly 1 index.
void plot_gaussian(matplot::axes_handle ax,
                   const brew::distributions::Gaussian& g,
                   const std::vector<int>& plt_inds,
                   const Color& color,
                   double num_std = 2.0);

/// Plot a 2D Gaussian with mean marker and filled covariance ellipse.
/// plt_inds must contain exactly 2 indices.
void plot_gaussian_2d(matplot::axes_handle ax,
                      const brew::distributions::Gaussian& g,
                      const std::vector<int>& plt_inds,
                      const Color& color,
                      double num_std = 2.0,
                      float alpha = 0.3f);

/// Plot a 3D Gaussian with mean marker and covariance ellipsoid surface.
/// plt_inds must contain exactly 3 indices.
void plot_gaussian_3d(matplot::axes_handle ax,
                      const brew::distributions::Gaussian& g,
                      const std::vector<int>& plt_inds,
                      const Color& color,
                      double num_std = 2.0,
                      float alpha = 0.3f,
                      float colormap_value = 0.0f);

/// Convenience: auto-dispatch based on plt_inds.size().
void plot_gaussian(matplot::axes_handle ax,
                   const brew::distributions::Gaussian& g,
                   const PlotOptions& opts);

/// Convenience: create figure, plot, save if output_file set.
void plot_gaussian(const brew::distributions::Gaussian& g,
                   const PlotOptions& opts);

} // namespace brew::plot_utils

