#include <brew/plot.hpp>
#include <brew/plot_utils/plot_gaussian.hpp>
#include <brew/plot_utils/plot_ggiw.hpp>
#include <brew/plot_utils/plot_mixture.hpp>
#include <brew/plot_utils/plot_trajectory_gaussian.hpp>
#include <brew/plot_utils/plot_trajectory_ggiw.hpp>

namespace brew {

void plot(const distributions::Gaussian& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_gaussian(g, opts);
}

void plot(const distributions::GGIW& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_ggiw(g, opts);
}

void plot(const distributions::TrajectoryGaussian& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_gaussian(g, opts);
}

void plot(const distributions::TrajectoryGGIW& g,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_trajectory_ggiw(g, opts);
}

void plot(const distributions::Mixture<distributions::Gaussian>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const distributions::Mixture<distributions::GGIW>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const distributions::Mixture<distributions::TrajectoryGaussian>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

void plot(const distributions::Mixture<distributions::TrajectoryGGIW>& mix,
          const plot_utils::PlotOptions& opts) {
    plot_utils::plot_mixture(mix, opts);
}

} // namespace brew
