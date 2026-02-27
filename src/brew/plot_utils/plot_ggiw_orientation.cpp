#include "brew/plot_utils/plot_ggiw_orientation.hpp"
#include "brew/plot_utils/plot_ggiw.hpp"
#include "brew/plot_utils/math_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace brew::plot_utils {

void plot_ggiw_orientation_2d(matplot::axes_handle ax,
                              const brew::models::GGIWOrientation& g,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence) {
    if (plt_inds.size() != 2) {
        throw std::invalid_argument("plot_ggiw_orientation_2d: plt_inds must have 2 elements");
    }

    // Draw the base GGIW ellipse + confidence interval
    plot_ggiw_2d(ax, g, plt_inds, color, confidence);

    // Draw principal axes as directional rays from center outward.
    // Each eigenvector is drawn from center to center + scale*eigvec,
    // producing an asymmetric cross that reveals the heading angle.
    if (g.basis().size() > 0 && g.has_eigenvalues()) {
        const double cx = g.mean()(plt_inds[0]);
        const double cy = g.mean()(plt_inds[1]);

        const Eigen::MatrixXd& basis = g.basis();
        const Eigen::MatrixXd& eig_mat = g.eigenvalues();
        const int d = g.extent_dim();

        for (int k = 0; k < std::min(d, 2); ++k) {
            double eigenval = eig_mat(k, k);
            double scale = std::sqrt(std::max(eigenval, 0.0));

            double ux = basis(0, k);
            double uy = basis(1, k);

            // Ray from center outward along the eigenvector direction
            std::vector<double> lx = { cx, cx + scale * ux };
            std::vector<double> ly = { cy, cy + scale * uy };

            auto line = ax->plot(lx, ly);
            line->color(color);
            line->line_width(2.0f);
        }
    }
}

void plot_ggiw_orientation(matplot::axes_handle ax,
                           const brew::models::GGIWOrientation& g,
                           const PlotOptions& opts) {
    switch (opts.plt_inds.size()) {
        case 2:
            plot_ggiw_orientation_2d(ax, g, opts.plt_inds, opts.color, opts.confidence);
            break;
        default:
            throw std::invalid_argument("plot_ggiw_orientation: plt_inds must have 2 elements");
    }
}

void plot_ggiw_orientation(const brew::models::GGIWOrientation& g,
                           const PlotOptions& opts) {
    auto fig = matplot::figure(true);
    fig->width(opts.width);
    fig->height(opts.height);
    auto ax = fig->current_axes();

    plot_ggiw_orientation(ax, g, opts);

    if (!opts.output_file.empty()) {
        save_figure(fig, opts.output_file);
    }
}

} // namespace brew::plot_utils
