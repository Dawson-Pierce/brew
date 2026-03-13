#include "brew/plot_utils/plot_point_cloud.hpp"
#include <stdexcept>

namespace brew::plot_utils {

void plot_point_cloud_2d(matplot::axes_handle ax,
                         const brew::template_matching::PointCloud& cloud,
                         const std::vector<int>& plt_inds,
                         const Color& color,
                         float marker_size) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_point_cloud_2d: plt_inds must have 2 elements");
    }

    const int N = cloud.num_points();
    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = cloud.points()(plt_inds[0], i);
        y[i] = cloud.points()(plt_inds[1], i);
    }

    ax->hold(true);
    auto s = ax->scatter(x, y);
    s->color(color);
    s->marker_size(marker_size);
    s->marker_face_color(color);
    s->marker_face(true);
}

void plot_point_cloud_3d(matplot::axes_handle ax,
                         const brew::template_matching::PointCloud& cloud,
                         const std::vector<int>& plt_inds,
                         const Color& color,
                         float marker_size) {
    if (plt_inds.size() != 3) {
        throw std::invalid_argument("plot_point_cloud_3d: plt_inds must have 3 elements");
    }

    const int N = cloud.num_points();
    std::vector<double> x(N), y(N), z(N);
    for (int i = 0; i < N; ++i) {
        x[i] = cloud.points()(plt_inds[0], i);
        y[i] = cloud.points()(plt_inds[1], i);
        z[i] = cloud.points()(plt_inds[2], i);
    }

    ax->hold(true);
    auto s = ax->plot3(x, y, z, ".");
    s->color(color);
    s->marker(matplot::line_spec::marker_style::circle);
    s->marker_size(marker_size);
}

void plot_point_cloud(matplot::axes_handle ax,
                      const brew::template_matching::PointCloud& cloud,
                      const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 2:
            plot_point_cloud_2d(ax, cloud, opts.plt_inds, opts.color);
            break;
        case 3:
            plot_point_cloud_3d(ax, cloud, opts.plt_inds, opts.color);
            break;
        default:
            throw std::invalid_argument("plot_point_cloud: plt_inds must have 2 or 3 elements");
    }
}

void plot_point_cloud(const brew::template_matching::PointCloud& cloud,
                      const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_point_cloud(ax, cloud, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils
