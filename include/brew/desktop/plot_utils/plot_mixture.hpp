#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include "brew/desktop/plot_utils/plot_trajectory_gaussian.hpp"
#include "brew/desktop/plot_utils/plot_trajectory_ggiw.hpp"
#include "brew/desktop/plot_utils/plot_trajectory_ggiw_orientation.hpp"
#include "brew/desktop/plot_utils/color_palette.hpp"
#include "brew/desktop/plot_utils/math_utils.hpp"
#include <brew/shared/mixture.hpp>
#include <brew/gaussian/gaussian_model.hpp>
#include <brew/ggiw/ggiw_model.hpp>
#include <brew/ggiw_orientation/ggiw_orientation_model.hpp>
#include <brew/trajectory_gaussian/trajectory_gaussian_model.hpp>
#include <brew/trajectory_ggiw/trajectory_ggiw_model.hpp>
#include <brew/trajectory_ggiw_orientation/trajectory_ggiw_orientation_model.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// Plot a Gaussian mixture. Each component gets a distinct color from the MATLAB lines palette.
void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::Gaussian<>>& mix,
                  const PlotOptions& opts);

/// Plot a GGIW mixture.
void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::GGIW<>>& mix,
                  const PlotOptions& opts);

/// Plot a GGIWOrientation mixture (with principal axes).
void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::GGIWOrientation<>>& mix,
                  const PlotOptions& opts);

namespace detail {

inline std::vector<std::vector<double>> to_colormap_argb(const std::vector<Color>& colors) {
    std::vector<std::vector<double>> cmap;
    cmap.reserve(colors.size());
    for (const auto& c : colors)
        cmap.push_back({c[1], c[2], c[3]});  // RGB from ARGB
    return cmap;
}

} // namespace detail

/// Plot a TrajectoryGaussian mixture (color order reversed to match MATLAB).
template <int MaxWindow>
void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::TrajectoryGaussian<MaxWindow>>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    // Reverse color order to match MATLAB behavior
    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[n - 1 - i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(n - 1 - i) / static_cast<float>(n - 1) : 0.0f;
        plot_trajectory_gaussian(ax, mix.component(i), comp_opts);
    }

    if (opts.plt_inds.size() == 3) {
        ax->colormap(detail::to_colormap_argb(colors));
    }
}

/// Plot a TrajectoryGGIW mixture (color order reversed to match MATLAB).
template <int MaxWindow>
void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::TrajectoryGGIW<MaxWindow>>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    // Reverse color order to match MATLAB behavior
    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[n - 1 - i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(n - 1 - i) / static_cast<float>(n - 1) : 0.0f;
        plot_trajectory_ggiw(ax, mix.component(i), comp_opts);
    }

    if (opts.plt_inds.size() == 3) {
        ax->colormap(detail::to_colormap_argb(colors));
    }
}

/// Plot a TrajectoryGGIWOrientation mixture (color order reversed, with principal axes).
template <int MaxWindow>
void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::TrajectoryGGIWOrientation<MaxWindow>>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    // Reverse color order to match MATLAB behavior
    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[n - 1 - i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(n - 1 - i) / static_cast<float>(n - 1) : 0.0f;
        plot_trajectory_ggiw_orientation(ax, mix.component(i), comp_opts);
    }
}

/// Convenience overloads: create figure, plot, save.
void plot_mixture(const brew::models::Mixture<brew::models::Gaussian<>>& mix,
                  const PlotOptions& opts);
void plot_mixture(const brew::models::Mixture<brew::models::GGIW<>>& mix,
                  const PlotOptions& opts);
void plot_mixture(const brew::models::Mixture<brew::models::GGIWOrientation<>>& mix,
                  const PlotOptions& opts);

template <int MaxWindow>
void plot_mixture(const brew::models::Mixture<brew::models::TrajectoryGaussian<MaxWindow>>& mix,
                  const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_mixture(ax, mix, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

template <int MaxWindow>
void plot_mixture(const brew::models::Mixture<brew::models::TrajectoryGGIW<MaxWindow>>& mix,
                  const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_mixture(ax, mix, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

template <int MaxWindow>
void plot_mixture(const brew::models::Mixture<brew::models::TrajectoryGGIWOrientation<MaxWindow>>& mix,
                  const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_mixture(ax, mix, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils

