#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include "brew/desktop/plot_utils/math_utils.hpp"
#include <brew/shared/trajectory.hpp>
#include <brew/gaussian/gaussian_model.hpp>
#include <matplot/matplot.h>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace brew::plot_utils {

/// Plot 1D TrajectoryGaussian: state value vs time index + window overlay.
/// plt_inds must contain exactly 1 index.
template <int MaxWindow>
void plot_trajectory_gaussian_1d(matplot::axes_handle ax,
                                  const brew::models::Trajectory<brew::models::Gaussian<>, MaxWindow>& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color) {
    if (plt_inds.size() != 1) {
        throw std::invalid_argument("plot_trajectory_gaussian_1d: plt_inds must have 1 element");
    }

    const int idx = plt_inds[0];
    const auto& sh = tg.state_history();
    const int T_hist = static_cast<int>(sh.size());
    const bool append_current = tg.window_size() > 0
        && (T_hist == 0 || !sh.back().mean().isApprox(tg.current().mean()));
    const int T = T_hist + (append_current ? 1 : 0);
    Eigen::MatrixXd states(tg.state_dim, T);
    for (int i = 0; i < T_hist; ++i) states.col(i) = sh[i].mean();
    if (append_current) states.col(T - 1) = tg.current().mean();
    const int t0 = 0;

    // Time indices
    std::vector<double> time_inds(T);
    std::vector<double> vals(T);
    for (int t = 0; t < T; ++t) {
        time_inds[t] = static_cast<double>(t0 + t);
        vals[t] = states(idx, t);
    }

    ax->hold(true);
    auto line = ax->plot(time_inds, vals);
    line->color(color);
    line->line_width(1.5f);

    // Window overlay: highlight the trajectory window region
    if (T > 1) {
        double t_start = static_cast<double>(t0);
        double t_end = static_cast<double>(t0 + T - 1);
        double y_min = *std::min_element(vals.begin(), vals.end());
        double y_max = *std::max_element(vals.begin(), vals.end());
        double margin = (y_max - y_min) * 0.1;
        if (margin < 1e-6) margin = 1.0;

        std::vector<double> wx = {t_start, t_end, t_end, t_start, t_start};
        std::vector<double> wy = {y_min - margin, y_min - margin,
                                   y_max + margin, y_max + margin, y_min - margin};
        auto fill_h = ax->fill(wx, wy);
        fill_h->color({0.9f, color[1], color[2], color[3]});
    }
}

/// Plot 2D TrajectoryGaussian: trajectory line from rearrange_states + window overlay.
/// plt_inds must contain exactly 2 indices.
template <int MaxWindow>
void plot_trajectory_gaussian_2d(matplot::axes_handle ax,
                                  const brew::models::Trajectory<brew::models::Gaussian<>, MaxWindow>& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_trajectory_gaussian_2d: plt_inds must have 2 elements");
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

    // End marker
    std::vector<double> ex = {x.back()};
    std::vector<double> ey = {y.back()};
    auto end_marker = ax->plot(ex, ey);
    end_marker->marker(matplot::line_spec::marker_style::square);
    end_marker->color(color);
    end_marker->marker_size(8.0f);
}

/// Plot 3D TrajectoryGaussian: plot3 trajectory + window.
/// plt_inds must contain exactly 3 indices.
template <int MaxWindow>
void plot_trajectory_gaussian_3d(matplot::axes_handle ax,
                                  const brew::models::Trajectory<brew::models::Gaussian<>, MaxWindow>& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_trajectory_gaussian_3d: plt_inds must have 3 elements");
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

    // End marker
    std::vector<double> ex_v = {x.back()};
    std::vector<double> ey_v = {y.back()};
    std::vector<double> ez_v = {z.back()};
    auto end_marker = ax->plot3(ex_v, ey_v, ez_v);
    end_marker->marker(matplot::line_spec::marker_style::square);
    end_marker->color(color);
    end_marker->marker_size(8.0f);
}

/// Convenience: auto-dispatch based on plt_inds.size().
template <int MaxWindow>
void plot_trajectory_gaussian(matplot::axes_handle ax,
                               const brew::models::Trajectory<brew::models::Gaussian<>, MaxWindow>& tg,
                               const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 1:
            plot_trajectory_gaussian_1d(ax, tg, opts.plt_inds, opts.color);
            break;
        case 2:
            plot_trajectory_gaussian_2d(ax, tg, opts.plt_inds, opts.color);
            break;
        case 3:
            plot_trajectory_gaussian_3d(ax, tg, opts.plt_inds, opts.color);
            break;
        default:
            throw std::invalid_argument("plot_trajectory_gaussian: plt_inds must have 1, 2, or 3 elements");
    }
}

/// Convenience: create figure, plot, save if output_file set.
template <int MaxWindow>
void plot_trajectory_gaussian(const brew::models::Trajectory<brew::models::Gaussian<>, MaxWindow>& tg,
                               const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_trajectory_gaussian(ax, tg, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils

