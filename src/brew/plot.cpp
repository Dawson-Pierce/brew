#include <brew/plot.hpp>
#include <brew/plot_utils/plot_gaussian.hpp>
#include <brew/plot_utils/plot_ggiw.hpp>
#include <brew/plot_utils/plot_mixture.hpp>
#include <brew/plot_utils/plot_trajectory_gaussian.hpp>
#include <brew/plot_utils/plot_trajectory_ggiw.hpp>

namespace brew {

void plot(const models::Gaussian& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_gaussian(g, opts);
}

void plot(const models::GGIW& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_ggiw(g, opts);
}

void plot(const models::TrajectoryGaussian& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_gaussian(g, opts);
}

void plot(const models::TrajectoryGGIW& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_ggiw(g, opts);
}

void plot(const models::Mixture<models::Gaussian>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const models::Mixture<models::GGIW>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const models::Mixture<models::TrajectoryGaussian>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const models::Mixture<models::TrajectoryGGIW>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

} // namespace brew
