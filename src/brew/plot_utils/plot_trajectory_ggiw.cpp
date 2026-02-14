#include "brew/plot_utils/plot_trajectory_ggiw.hpp"
#include "brew/plot_utils/plot_ggiw.hpp"
#include "brew/plot_utils/math_utils.hpp"
#include <stdexcept>

namespace brew::plot_utils {

void plot_trajectory_ggiw_2d(matplot::axes_handle ax,
                              const brew::distributions::TrajectoryGGIW& tg,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_trajectory_ggiw_2d: plt_inds must have 2 elements");
    }

    // Use full mean_history when available (shows entire trajectory, not just window)
    Eigen::MatrixXd states = (tg.mean_history().cols() > 0)
        ? tg.mean_history() : tg.rearrange_states();
    const int T = static_cast<int>(states.cols());

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

    // Draw GGIW extent ellipses at the last state
    auto last_mean = tg.get_last_state();
    auto last_cov = tg.get_last_cov();
    brew::distributions::GGIW last_ggiw(
        last_mean, last_cov,
        tg.alpha(), tg.beta(),
        tg.v(), tg.V());

    plot_ggiw_2d(ax, last_ggiw, plt_inds, color, confidence);
}

void plot_trajectory_ggiw_3d(matplot::axes_handle ax,
                              const brew::distributions::TrajectoryGGIW& tg,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence,
                              float alpha) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_trajectory_ggiw_3d: plt_inds must have 3 elements");
    }

    // Use full mean_history when available (shows entire trajectory, not just window)
    Eigen::MatrixXd states = (tg.mean_history().cols() > 0)
        ? tg.mean_history() : tg.rearrange_states();
    const int T = static_cast<int>(states.cols());

    std::vector<double> x(T), y(T), z(T);
    for (int t = 0; t < T; ++t) {
        x[t] = states(plt_inds[0], t);
        y[t] = states(plt_inds[1], t);
        z[t] = states(plt_inds[2], t);
    }

    ax->hold(true);
    auto line = ax->plot3(x, y, z);
    line->color(color);
    line->line_width(1.5f);

    // Start marker
    std::vector<double> sx = {x.front()};
    std::vector<double> sy = {y.front()};
    std::vector<double> sz = {z.front()};
    auto start_marker = ax->plot3(sx, sy, sz);
    start_marker->marker(matplot::line_spec::marker_style::circle);
    start_marker->color(color);
    start_marker->marker_size(8.0f);

    // Draw GGIW extent ellipsoids at the last state
    auto last_mean = tg.get_last_state();
    auto last_cov = tg.get_last_cov();
    brew::distributions::GGIW last_ggiw(
        last_mean, last_cov,
        tg.alpha(), tg.beta(),
        tg.v(), tg.V());

    plot_ggiw_3d(ax, last_ggiw, plt_inds, color, confidence, alpha);
}

void plot_trajectory_ggiw(matplot::axes_handle ax,
                           const brew::distributions::TrajectoryGGIW& tg,
                           const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 2:
            plot_trajectory_ggiw_2d(ax, tg, opts.plt_inds, opts.color, opts.confidence);
            break;
        case 3:
            plot_trajectory_ggiw_3d(ax, tg, opts.plt_inds, opts.color, opts.confidence, opts.alpha);
            break;
        default:
            throw std::invalid_argument("plot_trajectory_ggiw: plt_inds must have 2 or 3 elements");
    }
}

void plot_trajectory_ggiw(const brew::distributions::TrajectoryGGIW& tg,
                           const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_trajectory_ggiw(ax, tg, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils

