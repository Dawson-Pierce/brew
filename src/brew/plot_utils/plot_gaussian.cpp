#include "brew/plot_utils/plot_gaussian.hpp"
#include "brew/plot_utils/math_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <numbers>

namespace brew::plot_utils {

void plot_gaussian(matplot::axes_handle ax,
                   const brew::models::Gaussian& g,
                   const std::vector<int>& plt_inds,
                   const Color& color,
                   double num_std) {
    if (plt_inds.size() != 1) {
        throw std::invalid_argument("plot_gaussian 1D: plt_inds must have 1 element");
    }
    const int idx = plt_inds[0];
    const double mu = g.mean()(idx);
    const double sigma = std::sqrt(g.covariance()(idx, idx));

    // Generate x range: mu +/- num_std*sigma
    auto x = linspace(mu - num_std * sigma * 2.0, mu + num_std * sigma * 2.0, 200);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = normpdf(x[i], mu, sigma);
    }

    auto line = ax->plot(x, y);
    line->color(color);
    line->line_width(1.5f);

    // Mean marker
    ax->hold(true);
    std::vector<double> mx_vec = {mu};
    std::vector<double> my_vec = {normpdf(mu, mu, sigma)};
    auto marker = ax->plot(mx_vec, my_vec);
    marker->marker(matplot::line_spec::marker_style::asterisk);
    marker->color(color);
    marker->marker_size(10.0f);
}

void plot_gaussian_2d(matplot::axes_handle ax,
                      const brew::models::Gaussian& g,
                      const std::vector<int>& plt_inds,
                      const Color& color,
                      double num_std,
                      float alpha) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_gaussian_2d: plt_inds must have 2 elements");
    }

    // Generate ellipse points
    auto pts = generate_ellipse_points(g.mean(), g.covariance(), plt_inds, num_std);

    // Extract x, y columns
    std::vector<double> x(pts.rows()), y(pts.rows());
    for (int i = 0; i < pts.rows(); ++i) {
        x[i] = pts(i, 0);
        y[i] = pts(i, 1);
    }

    // Filled ellipse
    ax->hold(true);
    auto fill_h = ax->fill(x, y);
    fill_h->color({alpha, color[1], color[2], color[3]});
    fill_h->line_width(1.0f);

    // Mean marker
    const double mx = g.mean()(plt_inds[0]);
    const double my = g.mean()(plt_inds[1]);
    std::vector<double> mx_vec = {mx};
    std::vector<double> my_vec = {my};
    auto marker = ax->plot(mx_vec, my_vec);
    marker->marker(matplot::line_spec::marker_style::asterisk);
    marker->color(color);
    marker->marker_size(10.0f);
}

void plot_gaussian_3d(matplot::axes_handle ax,
                      const brew::models::Gaussian& g,
                      const std::vector<int>& plt_inds,
                      const Color& color,
                      double num_std,
                      float alpha,
                      float colormap_value) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_gaussian_3d: plt_inds must have 3 elements");
    }

    auto mesh = generate_ellipsoid_mesh(g.mean(), g.covariance(), plt_inds, num_std);

    // Convert to vectors of vectors for surf
    auto X = eigen_to_mat(mesh.X);
    auto Y = eigen_to_mat(mesh.Y);
    auto Z = eigen_to_mat(mesh.Z);

    // Uniform CData so the surface uses a single color from the colormap
    auto C = X;
    for (auto& row : C) std::fill(row.begin(), row.end(), static_cast<double>(colormap_value));

    auto surf = ax->surf(X, Y, Z, C);
    surf->face_alpha(alpha);
    surf->edge_color({0.0f, 0.0f, 0.0f, 0.0f});
    ax->colormap(std::vector<std::vector<double>>{{color[1], color[2], color[3]}});

    // Mean marker
    ax->hold(true);
    std::vector<double> px = {g.mean()(plt_inds[0])};
    std::vector<double> py = {g.mean()(plt_inds[1])};
    std::vector<double> pz = {g.mean()(plt_inds[2])};
    auto marker = ax->plot3(px, py, pz);
    marker->marker(matplot::line_spec::marker_style::asterisk);
    marker->color(color);
    marker->marker_size(10.0f);
}

void plot_gaussian(matplot::axes_handle ax,
                   const brew::models::Gaussian& g,
                   const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 1:
            plot_gaussian(ax, g, opts.plt_inds, opts.color, opts.num_std);
            break;
        case 2:
            plot_gaussian_2d(ax, g, opts.plt_inds, opts.color, opts.num_std, opts.alpha);
            break;
        case 3:
            plot_gaussian_3d(ax, g, opts.plt_inds, opts.color, opts.num_std, opts.alpha, opts.colormap_value);
            break;
        default:
            throw std::invalid_argument("plot_gaussian: plt_inds must have 1, 2, or 3 elements");
    }
}

void plot_gaussian(const brew::models::Gaussian& g,
                   const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_gaussian(ax, g, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils

