#include "brew/desktop/plot_utils/plot_trajectory_ggiw_orientation.hpp"
#include "brew/desktop/plot_utils/plot_ggiw_orientation.hpp"
#include "brew/desktop/plot_utils/math_utils.hpp"
#include "brew/core/models/ggiw_orientation.hpp"
#include <stdexcept>
#include <cmath>

namespace brew::plot_utils {

void plot_trajectory_ggiw_orientation_2d(matplot::axes_handle ax,
                                         const brew::models::Trajectory<brew::models::GGIWOrientation<>>& tg,
                                         const std::vector<int>& plt_inds,
                                         const Color& color,
                                         double confidence) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_trajectory_ggiw_orientation_2d: plt_inds must have 2 elements");
    }

    const auto& sh = tg.state_history();
    const int T_hist = static_cast<int>(sh.size());
    const bool append_current = tg.window_size() > 0
        && (T_hist == 0 || !sh.back().mean().isApprox(tg.current().mean()));
    const int T = T_hist + (append_current ? 1 : 0);
    Eigen::MatrixXd states(tg.state_dim, T);
    for (int i = 0; i < T_hist; ++i) states.col(i) = sh[i].mean();
    if (append_current) states.col(T - 1) = tg.current().mean();

    std::vector<double> x(T), y(T);
    for (int t = 0; t < T; ++t) {
        x[t] = states(plt_inds[0], t);
        y[t] = states(plt_inds[1], t);
    }

    ax->hold(true);
    auto line = ax->plot(x, y);
    line->color(color);
    line->line_width(1.5f);

    // Start marker
    std::vector<double> sx = {x.front()};
    std::vector<double> sy = {y.front()};
    auto start_marker = ax->plot(sx, sy);
    start_marker->marker(matplot::line_spec::marker_style::circle);
    start_marker->color(color);
    start_marker->marker_size(8.0f);

    // Draw GGIW orientation extent ellipses + principal axes at the last state
    auto last_mean = tg.get_last_state();
    auto last_cov = tg.get_last_cov();
    brew::models::GGIWOrientation<> last_ggiw(
        tg.current().alpha(), tg.current().beta(),
        last_mean, last_cov,
        tg.current().v(), tg.current().V());

    // Copy basis from trajectory if available
    if (tg.current().basis().size() > 0) {
        last_ggiw.basis() = tg.current().basis();
        last_ggiw.eigenvalues() = tg.current().eigenvalues();
    }

    plot_ggiw_orientation_2d(ax, last_ggiw, plt_inds, color, confidence);
}

void plot_trajectory_ggiw_orientation(matplot::axes_handle ax,
                                      const brew::models::Trajectory<brew::models::GGIWOrientation<>>& tg,
                                      const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 2:
            plot_trajectory_ggiw_orientation_2d(ax, tg, opts.plt_inds, opts.color, opts.confidence);
            break;
        default:
            throw std::invalid_argument("plot_trajectory_ggiw_orientation: plt_inds must have 2 elements");
    }
}

void plot_trajectory_ggiw_orientation(const brew::models::Trajectory<brew::models::GGIWOrientation<>>& tg,
                                      const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_trajectory_ggiw_orientation(ax, tg, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils
