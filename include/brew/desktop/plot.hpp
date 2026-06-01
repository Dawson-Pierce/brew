#pragma once

#include <brew/gaussian/gaussian_model.hpp>
#include <brew/ggiw/ggiw_model.hpp>
#include <brew/shared/mixture.hpp>
#include <brew/trajectory_gaussian/trajectory_gaussian_model.hpp>
#include <brew/trajectory_ggiw/trajectory_ggiw_model.hpp>
#include <brew/template_matching/point_cloud.hpp>
#include <brew/desktop/plot_utils/plot_options.hpp>
#include <brew/desktop/plot_utils/plot_mixture.hpp>
#include <brew/desktop/plot_utils/plot_trajectory_gaussian.hpp>
#include <brew/desktop/plot_utils/plot_trajectory_ggiw.hpp>

namespace brew {

// Convenience wrappers at the top-level brew namespace.
// These forward to brew::plot_utils::* so callers don't need to reference the sub-namespace.
void plot(const models::Gaussian<>& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::GGIW<>& g,
          const plot_utils::PlotOptions& opts);

template <int MaxWindow>
void plot(const models::TrajectoryGaussian<MaxWindow>& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_gaussian(g, opts);
}

template <int MaxWindow>
void plot(const models::TrajectoryGGIW<MaxWindow>& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_ggiw(g, opts);
}

void plot(const models::Mixture<models::Gaussian<>>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::GGIW<>>& mix,
          const plot_utils::PlotOptions& opts);

template <int MaxWindow>
void plot(const models::Mixture<models::TrajectoryGaussian<MaxWindow>>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

template <int MaxWindow>
void plot(const models::Mixture<models::TrajectoryGGIW<MaxWindow>>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const template_matching::PointCloud& cloud,
          const plot_utils::PlotOptions& opts);

} // namespace brew
