#pragma once

#include <brew/distributions/gaussian.hpp>
#include <brew/distributions/ggiw.hpp>
#include <brew/distributions/mixture.hpp>
#include <brew/distributions/trajectory_gaussian.hpp>
#include <brew/distributions/trajectory_ggiw.hpp>
#include <brew/plot_utils/plot_options.hpp>

namespace brew {

// Convenience wrappers at the top-level brew namespace.
// These forward to brew::plot_utils::* so callers don't need to reference the sub-namespace.
void plot(const distributions::Gaussian& g,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::GGIW& g,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::TrajectoryGaussian& g,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::TrajectoryGGIW& g,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::Mixture<distributions::Gaussian>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::Mixture<distributions::GGIW>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::Mixture<distributions::TrajectoryGaussian>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const distributions::Mixture<distributions::TrajectoryGGIW>& mix,
          const plot_utils::PlotOptions& opts);

} // namespace brew
