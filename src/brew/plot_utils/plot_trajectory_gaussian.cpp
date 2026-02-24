#include "brew/plot_utils/plot_trajectory_gaussian.hpp"
#include "brew/plot_utils/math_utils.hpp"
#include <stdexcept>
#include <algorithm>

namespace brew::plot_utils {

void plot_trajectory_gaussian_1d(matplot::axes_handle ax,
                                  const brew::models::TrajectoryGaussian& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color) {
    if (plt_inds.size() != 1) {
        throw std::invalid_argument("plot_trajectory_gaussian_1d: plt_inds must have 1 element");
    }

    const int idx = plt_inds[0];
    // Use full mean_history when available (shows entire trajectory, not just window)
    Eigen::MatrixXd states = (tg.mean_history().cols() > 0)
        ? tg.mean_history() : tg.rearrange_states();
    const int T = static_cast<int>(states.cols());
    const int t0 = tg.init_idx;

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

void plot_trajectory_gaussian_2d(matplot::axes_handle ax,
                                  const brew::models::TrajectoryGaussian& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_trajectory_gaussian_2d: plt_inds must have 2 elements");
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

    // End marker
    std::vector<double> ex = {x.back()};
    std::vector<double> ey = {y.back()};
    auto end_marker = ax->plot(ex, ey);
    end_marker->marker(matplot::line_spec::marker_style::square);
    end_marker->color(color);
    end_marker->marker_size(8.0f);
}

void plot_trajectory_gaussian_3d(matplot::axes_handle ax,
                                  const brew::models::TrajectoryGaussian& tg,
                                  const std::vector<int>& plt_inds,
                                  const Color& color) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_trajectory_gaussian_3d: plt_inds must have 3 elements");
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

    // End marker
    std::vector<double> ex_v = {x.back()};
    std::vector<double> ey_v = {y.back()};
    std::vector<double> ez_v = {z.back()};
    auto end_marker = ax->plot3(ex_v, ey_v, ez_v);
    end_marker->marker(matplot::line_spec::marker_style::square);
    end_marker->color(color);
    end_marker->marker_size(8.0f);
}

void plot_trajectory_gaussian(matplot::axes_handle ax,
                               const brew::models::TrajectoryGaussian& tg,
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

void plot_trajectory_gaussian(const brew::models::TrajectoryGaussian& tg,
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

