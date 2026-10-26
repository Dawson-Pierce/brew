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

void plot(const models::Gaussian<>& g,
          const plot_utils::PlotOptions& opts);
void plot(const models::GGIW<>& g,
          const plot_utils::PlotOptions& opts);

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::TrajectoryGaussian<Scalar, D>& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_gaussian(g, opts);
}

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::TrajectoryGGIW<Scalar, D, De>& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_ggiw(g, opts);
}

void plot(const models::Mixture<models::Gaussian<>>& mix,
          const plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::GGIW<>>& mix,
          const plot_utils::PlotOptions& opts);

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::Mixture<models::TrajectoryGaussian<Scalar, D>>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::Mixture<models::TrajectoryGGIW<Scalar, D, De>>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const template_matching::PointCloud& cloud,
          const plot_utils::PlotOptions& opts);

}
