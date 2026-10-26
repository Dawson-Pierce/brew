#include "brew/desktop/plot_utils/plot_mixture.hpp"
#include "brew/desktop/plot_utils/plot_gaussian.hpp"
#include "brew/desktop/plot_utils/plot_ggiw.hpp"
#include "brew/desktop/plot_utils/plot_ggiw_orientation.hpp"
#include "brew/desktop/plot_utils/color_palette.hpp"

namespace brew::plot_utils {

void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::Gaussian<>>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
        plot_gaussian(ax, mix.component(i), comp_opts);
    }

    if (opts.plt_inds.size() == 3) {
        ax->colormap(detail::to_colormap_argb(colors));
    }
}

void plot_mixture(const brew::models::Mixture<brew::models::Gaussian<>>& mix,
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

void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::GGIW<>>& mix,
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
        ax->colormap(detail::to_colormap_argb(colors));
    }
}

void plot_mixture(const brew::models::Mixture<brew::models::GGIW<>>& mix,
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

void plot_mixture(matplot::axes_handle ax,
                  const brew::models::Mixture<brew::models::GGIWOrientation<>>& mix,
                  const PlotOptions& opts) {
    const int n = static_cast<int>(mix.size());
    auto colors = lines_colors(n);

    for (int i = 0; i < n; ++i) {
        PlotOptions comp_opts = opts;
        comp_opts.color = colors[i];
        comp_opts.colormap_value = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
        plot_ggiw_orientation(ax, mix.component(i), comp_opts);
    }
}

void plot_mixture(const brew::models::Mixture<brew::models::GGIWOrientation<>>& mix,
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

}
