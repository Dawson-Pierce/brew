#include <brew/desktop/plot.hpp>
#include <brew/desktop/plot_utils/plot_gaussian.hpp>
#include <brew/desktop/plot_utils/plot_ggiw.hpp>
#include <brew/desktop/plot_utils/plot_mixture.hpp>
#include <brew/desktop/plot_utils/plot_point_cloud.hpp>

namespace brew {

void plot(const models::Gaussian<>& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_gaussian(g, opts);
}

void plot(const models::GGIW<>& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_ggiw(g, opts);
}

void plot(const models::Mixture<models::Gaussian<>>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const models::Mixture<models::GGIW<>>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const template_matching::PointCloud& cloud,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_point_cloud(cloud, opts);
}

} // namespace brew
