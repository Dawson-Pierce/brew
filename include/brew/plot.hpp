#pragma once

#include <brew/models/gaussian.hpp>
#include <brew/models/ggiw.hpp>
#include <brew/models/mixture.hpp>
#include <brew/models/trajectory_gaussian.hpp>
#include <brew/models/trajectory_ggiw.hpp>
#include <brew/plot_utils/plot_options.hpp>

namespace brew {

// Convenience wrappers at the top-level brew namespace.
// These forward to brew::plot_utils::* so callers don't need to reference the sub-namespace.
void plot(const models::Gaussian& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::GGIW& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::TrajectoryGaussian& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::TrajectoryGGIW& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::Gaussian>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::GGIW>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::TrajectoryGaussian>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::TrajectoryGGIW>& mix,
          const plot_utils::PlotOptions& opts);

} // namespace brew
