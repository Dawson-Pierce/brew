#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <brew/distributions/mixture.hpp>
#include <brew/distributions/gaussian.hpp>
#include <brew/distributions/ggiw.hpp>
#include <brew/distributions/trajectory_gaussian.hpp>
#include <brew/distributions/trajectory_ggiw.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot a Gaussian mixture. Each component gets a distinct color from the MATLAB lines palette.
void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::Gaussian>& mix,
                  const PlotOptions& opts);

/// Plot a GGIW mixture.
void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::GGIW>& mix,
                  const PlotOptions& opts);

/// Plot a TrajectoryGaussian mixture (color order reversed to match MATLAB).
void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::TrajectoryGaussian>& mix,
                  const PlotOptions& opts);

/// Plot a TrajectoryGGIW mixture (color order reversed to match MATLAB).
void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::TrajectoryGGIW>& mix,
                  const PlotOptions& opts);

/// Convenience overloads: create figure, plot, save.
void plot_mixture(const brew::distributions::Mixture<brew::distributions::Gaussian>& mix,
                  const PlotOptions& opts);
void plot_mixture(const brew::distributions::Mixture<brew::distributions::GGIW>& mix,
                  const PlotOptions& opts);
void plot_mixture(const brew::distributions::Mixture<brew::distributions::TrajectoryGaussian>& mix,
                  const PlotOptions& opts);
void plot_mixture(const brew::distributions::Mixture<brew::distributions::TrajectoryGGIW>& mix,
                  const PlotOptions& opts);

} // namespace brew::plot_utils

