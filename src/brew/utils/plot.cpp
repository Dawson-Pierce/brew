#include "brew/utils/plot.hpp"
#include "brew/utils/plot_utils/plot_gaussian.hpp"
#include "brew/utils/plot_utils/plot_ggiw.hpp"
#include "brew/utils/plot_utils/plot_mixture.hpp"
#include "brew/utils/plot_utils/plot_point_cloud.hpp"

namespace brew::utils {

void plot(const models::Gaussian<>& g,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_gaussian(g, opts);
}

void plot(const models::GGIW<>& g,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_ggiw(g, opts);
}

void plot(const models::Mixture<models::Gaussian<>>& mix,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_mixture(mix, opts);
}

void plot(const models::Mixture<models::GGIW<>>& mix,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_mixture(mix, opts);
}

void plot(const utils::registration::PointCloud& cloud,
          const utils::plot_utils::PlotOptions& opts) {
    utils::plot_utils::plot_point_cloud(cloud, opts);
}

}
