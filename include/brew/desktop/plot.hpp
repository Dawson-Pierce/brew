#pragma once

#include <brew/core/models/gaussian.hpp>
#include <brew/core/models/ggiw.hpp>
#include <brew/core/models/mixture.hpp>
#include <brew/core/models/trajectory.hpp>
#include <brew/core/template_matching/point_cloud.hpp>
#include <brew/desktop/plot_utils/plot_options.hpp>

namespace brew {

// Convenience wrappers at the top-level brew namespace.
// These forward to brew::plot_utils::* so callers don't need to reference the sub-namespace.
void plot(const models::Gaussian<>& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::GGIW<>& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::Trajectory<models::Gaussian<>>& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::Trajectory<models::GGIW<>>& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::Gaussian<>>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::GGIW<>>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::Trajectory<models::Gaussian<>>>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::Trajectory<models::GGIW<>>>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const template_matching::PointCloud& cloud,
          const plot_utils::PlotOptions& opts);

} // namespace brew
