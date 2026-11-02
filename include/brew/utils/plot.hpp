#pragma once

#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/trajectory_ggiw.hpp"
#include "brew/utils/registration/point_cloud.hpp"
#include "brew/utils/plot_utils/plot_options.hpp"
#include "brew/utils/plot_utils/plot_mixture.hpp"
#include "brew/utils/plot_utils/plot_trajectory_gaussian.hpp"
#include "brew/utils/plot_utils/plot_trajectory_ggiw.hpp"

namespace brew::utils {

void plot(const models::Gaussian<>& g,
          const utils::plot_utils::PlotOptions& opts);
void plot(const models::GGIW<>& g,
          const utils::plot_utils::PlotOptions& opts);

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::TrajectoryGaussian<Scalar, D>& g,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_trajectory_gaussian(g, opts);
}

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::TrajectoryGGIW<Scalar, D, De>& g,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_trajectory_ggiw(g, opts);
}

void plot(const models::Mixture<models::Gaussian<>>& mix,
          const utils::plot_utils::PlotOptions& opts);
void plot(const models::Mixture<models::GGIW<>>& mix,
          const utils::plot_utils::PlotOptions& opts);

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::Mixture<models::TrajectoryGaussian<Scalar, D>>& mix,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_mixture(mix, opts);
}

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
void plot(const models::Mixture<models::TrajectoryGGIW<Scalar, D, De>>& mix,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_mixture(mix, opts);
}

void plot(const utils::registration::PointCloud& cloud,
          const utils::plot_utils::PlotOptions& opts);

}
