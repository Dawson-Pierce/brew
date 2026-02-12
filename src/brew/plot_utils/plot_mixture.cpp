#include "brew/plot_utils/plot_mixture.hpp"
#include "brew/plot_utils/plot_gaussian.hpp"
#include "brew/plot_utils/plot_ggiw.hpp"
#include "brew/plot_utils/plot_trajectory_gaussian.hpp"
#include "brew/plot_utils/plot_trajectory_ggiw.hpp"
#include "brew/plot_utils/color_palette.hpp"

namespace brew::plot_utils {

namespace {
std::vector<std::vector<double>> to_colormap(const std::vector<Color>& colors) {
    std::vector<std::vector<double>> cmap;
    cmap.reserve(colors.size());
    for (const auto& c : colors)
        cmap.push_back({c[1], c[2], c[3]});  // RGB from ARGB
    return cmap;
}
} // namespace

// --- Gaussian mixture ---

void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::Gaussian>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
        plot_gaussian(ax, mix.component(i), comp_opts);
    }

    // For 3D plots, set combined colormap so each component surface gets its own color
    if (opts.plt_inds.size() == 3) {
        ax->colormap(to_colormap(colors));
    }
}

void plot_mixture(const brew::distributions::Mixture<brew::distributions::Gaussian>& mix,
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

// --- GGIW mixture ---

void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::GGIW>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
        plot_ggiw(ax, mix.component(i), comp_opts);
    }

    if (opts.plt_inds.size() == 3) {
        ax->colormap(to_colormap(colors));
    }
}

void plot_mixture(const brew::distributions::Mixture<brew::distributions::GGIW>& mix,
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

// --- TrajectoryGaussian mixture (color order reversed) ---

void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::TrajectoryGaussian>& mix,
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
        ax->colormap(to_colormap(colors));
    }
}

void plot_mixture(const brew::distributions::Mixture<brew::distributions::TrajectoryGaussian>& mix,
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

// --- TrajectoryGGIW mixture (color order reversed) ---

void plot_mixture(matplot::axes_handle ax,
                  const brew::distributions::Mixture<brew::distributions::TrajectoryGGIW>& mix,
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
        ax->colormap(to_colormap(colors));
    }
}

void plot_mixture(const brew::distributions::Mixture<brew::distributions::TrajectoryGGIW>& mix,
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

