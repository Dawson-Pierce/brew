#include "brew/plot_utils/plot_ggiw.hpp"
#include "brew/plot_utils/math_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace brew::plot_utils {

void plot_ggiw_2d(matplot::axes_handle ax,
                  const brew::models::GGIW& g,
                  const std::vector<int>& plt_inds,
                  const Color& color,
                  double confidence) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_ggiw_2d: plt_inds must have 2 elements");
    }

    const int d = g.extent_dim();  // full IW dimension
    const double v = g.v();

    // Mean extent matrix: V / (v - d - 1)
    Eigen::MatrixXd mean_extent = g.V() / (v - d - 1.0);

    // Build center from kinematic mean at plt_inds
    Eigen::VectorXd center(2);
    center(0) = g.mean()(plt_inds[0]);
    center(1) = g.mean()(plt_inds[1]);

    // Map plt_inds to extent indices (assume extent is d-dimensional, use first 2)
    std::vector<int> extent_inds = {0, 1};

    // Mean extent ellipse (solid)
    auto pts_mean = generate_ellipse_points(center, mean_extent, extent_inds, 1.0);
    std::vector<double> xm(pts_mean.rows()), ym(pts_mean.rows());
    for (int i = 0; i < pts_mean.rows(); ++i) {
        xm[i] = pts_mean(i, 0);
        ym[i] = pts_mean(i, 1);
    }

    ax->hold(true);
    auto mean_line = ax->plot(xm, ym);
    mean_line->color(color);
    mean_line->line_width(2.0f);

    // Confidence interval ellipse (dashed)
    const int dof = static_cast<int>(plt_inds.size());
    const double ci_scale = std::sqrt(chi2inv(confidence, dof));
    auto pts_ci = generate_ellipse_points(center, mean_extent, extent_inds, ci_scale);
    std::vector<double> xci(pts_ci.rows()), yci(pts_ci.rows());
    for (int i = 0; i < pts_ci.rows(); ++i) {
        xci[i] = pts_ci(i, 0);
        yci[i] = pts_ci(i, 1);
    }

    auto ci_line = ax->plot(xci, yci, "--");
    ci_line->color(color);
    ci_line->line_width(1.5f);

    // Mean marker
    std::vector<double> cx = {center(0)};
    std::vector<double> cy = {center(1)};
    auto marker = ax->plot(cx, cy);
    marker->marker(matplot::line_spec::marker_style::asterisk);
    marker->color(color);
    marker->marker_size(10.0f);
}

void plot_ggiw_3d(matplot::axes_handle ax,
                  const brew::models::GGIW& g,
                  const std::vector<int>& plt_inds,
                  const Color& color,
                  double confidence,
                  float alpha,
                  float colormap_value) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_ggiw_3d: plt_inds must have 3 elements");
    }

    const int d = g.extent_dim();
    const double v = g.v();

    Eigen::MatrixXd mean_extent = g.V() / (v - d - 1.0);

    Eigen::VectorXd center(3);
    center(0) = g.mean()(plt_inds[0]);
    center(1) = g.mean()(plt_inds[1]);
    center(2) = g.mean()(plt_inds[2]);

    std::vector<int> extent_inds = {0, 1, 2};

    // Mean extent ellipsoid
    auto mesh_mean = generate_ellipsoid_mesh(center, mean_extent, extent_inds, 1.0);
    auto Xm = eigen_to_mat(mesh_mean.X);
    auto Ym = eigen_to_mat(mesh_mean.Y);
    auto Zm = eigen_to_mat(mesh_mean.Z);

    // Uniform CData so the surface uses a single color from the colormap
    auto Cm = Xm;
    for (auto& row : Cm) std::fill(row.begin(), row.end(), static_cast<double>(colormap_value));

    ax->hold(true);
    auto surf_mean = ax->surf(Xm, Ym, Zm, Cm);
    surf_mean->face_alpha(alpha);
    surf_mean->edge_color({0.0f, 0.0f, 0.0f, 0.0f});
    ax->colormap(std::vector<std::vector<double>>{{color[1], color[2], color[3]}});

    // Confidence extent ellipsoid
    const int dof = static_cast<int>(plt_inds.size());
    const double ci_scale = std::sqrt(chi2inv(confidence, dof));
    auto mesh_ci = generate_ellipsoid_mesh(center, mean_extent, extent_inds, ci_scale);
    auto Xci = eigen_to_mat(mesh_ci.X);
    auto Yci = eigen_to_mat(mesh_ci.Y);
    auto Zci = eigen_to_mat(mesh_ci.Z);

    auto Cci = Xci;
    for (auto& row : Cci) std::fill(row.begin(), row.end(), static_cast<double>(colormap_value));

    auto surf_ci = ax->surf(Xci, Yci, Zci, Cci);
    surf_ci->face_alpha(alpha * 0.5f);
    surf_ci->edge_color({0.0f, 0.0f, 0.0f, 0.0f});

    // Mean marker
    std::vector<double> px = {center(0)};
    std::vector<double> py = {center(1)};
    std::vector<double> pz = {center(2)};
    auto marker = ax->plot3(px, py, pz);
    marker->marker(matplot::line_spec::marker_style::asterisk);
    marker->color(color);
    marker->marker_size(10.0f);
}

void plot_ggiw(matplot::axes_handle ax,
               const brew::models::GGIW& g,
               const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 2:
            plot_ggiw_2d(ax, g, opts.plt_inds, opts.color, opts.confidence);
            break;
        case 3:
            plot_ggiw_3d(ax, g, opts.plt_inds, opts.color, opts.confidence, opts.alpha, opts.colormap_value);
            break;
        default:
            throw std::invalid_argument("plot_ggiw: plt_inds must have 2 or 3 elements");
    }
}

void plot_ggiw(const brew::models::GGIW& g,
               const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_ggiw(ax, g, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils

