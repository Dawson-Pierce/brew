#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include "brew/desktop/plot_utils/plot_ggiw.hpp"
#include "brew/desktop/plot_utils/math_utils.hpp"
#include <brew/core/models/trajectory.hpp>
#include <brew/core/models/ggiw.hpp>
#include <matplot/matplot.h>
#include <stdexcept>
#include <vector>

namespace brew::plot_utils {

/// Plot 2D TrajectoryGGIW: trajectory line + GGIW extent ellipses at last state.
/// plt_inds must contain exactly 2 indices.
template <int MaxWindow>
void plot_trajectory_ggiw_2d(matplot::axes_handle ax,
                              const brew::models::Trajectory<brew::models::GGIW<>, MaxWindow>& tg,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence = 0.99) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_trajectory_ggiw_2d: plt_inds must have 2 elements");
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

    // Draw GGIW extent ellipses at the last state
    auto last_mean = tg.get_last_state();
    auto last_cov = tg.get_last_cov();
    brew::models::GGIW<> last_ggiw(
        tg.current().alpha(), tg.current().beta(),
        last_mean, last_cov,
        tg.current().v(), tg.current().V());

    plot_ggiw_2d(ax, last_ggiw, plt_inds, color, confidence);
}

/// Plot 3D TrajectoryGGIW: plot3 trajectory + GGIW extent ellipsoids at last state.
/// plt_inds must contain exactly 3 indices.
template <int MaxWindow>
void plot_trajectory_ggiw_3d(matplot::axes_handle ax,
                              const brew::models::Trajectory<brew::models::GGIW<>, MaxWindow>& tg,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence = 0.99,
                              float alpha = 0.3f) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_trajectory_ggiw_3d: plt_inds must have 3 elements");
    }

    const auto& sh = tg.state_history();
    const int T_hist = static_cast<int>(sh.size());
    const bool append_current = tg.window_size() > 0
        && (T_hist == 0 || !sh.back().mean().isApprox(tg.current().mean()));
    const int T = T_hist + (append_current ? 1 : 0);
    Eigen::MatrixXd states(tg.state_dim, T);
    for (int i = 0; i < T_hist; ++i) states.col(i) = sh[i].mean();
    if (append_current) states.col(T - 1) = tg.current().mean();

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
    brew::models::GGIW<> last_ggiw(
        tg.current().alpha(), tg.current().beta(),
        last_mean, last_cov,
        tg.current().v(), tg.current().V());

    plot_ggiw_3d(ax, last_ggiw, plt_inds, color, confidence, alpha);
}

/// Convenience: auto-dispatch based on plt_inds.size().
template <int MaxWindow>
void plot_trajectory_ggiw(matplot::axes_handle ax,
                           const brew::models::Trajectory<brew::models::GGIW<>, MaxWindow>& tg,
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

/// Convenience: create figure, plot, save if output_file set.
template <int MaxWindow>
void plot_trajectory_ggiw(const brew::models::Trajectory<brew::models::GGIW<>, MaxWindow>& tg,
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

