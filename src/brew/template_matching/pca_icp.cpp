#include "brew/template_matching/pca_icp.hpp"
#include <Eigen/Eigenvalues>
#include <limits>

namespace brew::template_matching {

PcaIcp::PcaIcp(std::unique_ptr<IcpBase> inner_icp, const PointCloud& source_template)
    : inner_(std::move(inner_icp)) {
    src_axes_ = pca_axes(source_template.points());
    src_centroid_ = source_template.points().rowwise().mean();
}

Eigen::Matrix3d PcaIcp::pca_axes(const Eigen::MatrixXd& points) {
    Eigen::Vector3d mean = points.rowwise().mean();
    Eigen::MatrixXd centered = points.colwise() - mean;
    Eigen::Matrix3d cov = (centered * centered.transpose()) / (points.cols() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d axes;
    axes.col(0) = solver.eigenvectors().col(2);  // largest eigenvalue
    axes.col(1) = solver.eigenvectors().col(1);
    axes.col(2) = solver.eigenvectors().col(0);  // smallest

    // Ensure right-handed
    if (axes.determinant() < 0) {
        axes.col(2) = -axes.col(2);
    }
    return axes;
}

std::array<Eigen::Matrix3d, 8> PcaIcp::pca_candidates(
    const Eigen::Matrix3d& src_axes, const Eigen::Matrix3d& tgt_axes) {

    std::array<Eigen::Matrix3d, 8> candidates;
    int idx = 0;
    for (int s0 : {-1, 1}) {
        for (int s1 : {-1, 1}) {
            for (int s2 : {-1, 1}) {
                Eigen::Matrix3d flipped = src_axes;
                flipped.col(0) *= s0;
                flipped.col(1) *= s1;
                flipped.col(2) *= s2;
                if (flipped.determinant() < 0) {
                    flipped.col(2) = -flipped.col(2);
                }
                candidates[idx++] = tgt_axes * flipped.transpose();
            }
        }
    }
    return candidates;
}

IcpResult PcaIcp::align(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& /*t_init*/) const {

    // Compute target PCA
    Eigen::Matrix3d tgt_axes = pca_axes(target.points());
    Eigen::Vector3d tgt_centroid = target.points().rowwise().mean();

    // Generate 8 PCA rotation candidates
    auto candidates = pca_candidates(src_axes_, tgt_axes);

    // Score each candidate with a cheap RMSE on a subset of source points
    const int step = std::max(1, source.num_points() / 100);
    std::array<double, 8> costs{};
    for (int c = 0; c < 8; ++c) {
        Eigen::Vector3d t_cand = tgt_centroid - candidates[c] * src_centroid_;
        double cost = 0.0;
        for (int i = 0; i < source.num_points(); i += step) {
            Eigen::Vector3d p = candidates[c] * source.points().col(i) + t_cand;
            double min_sq = std::numeric_limits<double>::max();
            for (int j = 0; j < target.num_points(); ++j) {
                double sq = (p - target.points().col(j)).squaredNorm();
                if (sq < min_sq) min_sq = sq;
            }
            cost += min_sq;
        }
        costs[c] = cost;
    }

    // Sort candidates by cost and try the top few with full ICP,
    // keeping the result with the lowest RMSE. This resolves 180° ambiguities
    // that PCA scoring alone can't distinguish with partial measurements.
    std::array<int, 8> order = {0, 1, 2, 3, 4, 5, 6, 7};
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return costs[a] < costs[b]; });

    auto saved_params = inner_->params();
    auto cold_params = saved_params;
    cold_params.trim_fraction = 1.0;
    inner_->set_params(cold_params);

    const int n_tries = 3;  // try top 3 to resolve 180° ambiguity with partial views
    IcpResult best_result;
    best_result.rmse = std::numeric_limits<double>::max();
    for (int i = 0; i < n_tries; ++i) {
        Eigen::Matrix3d R_cand = candidates[order[i]];
        Eigen::Vector3d t_cand = tgt_centroid - R_cand * src_centroid_;
        auto result = inner_->align(source, target, R_cand, t_cand);
        if (result.rmse < best_result.rmse) {
            best_result = result;
        }
    }

    inner_->set_params(saved_params);
    return best_result;
}

double PcaIcp::log_likelihood(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t) const {
    return inner_->log_likelihood(source, target, R, t);
}

std::unique_ptr<IcpBase> PcaIcp::clone() const {
    auto cloned = std::make_unique<PcaIcp>(inner_->clone(), PointCloud(Eigen::MatrixXd(3, 0)));
    cloned->src_axes_ = src_axes_;
    cloned->src_centroid_ = src_centroid_;
    cloned->params_ = params_;
    return cloned;
}

} // namespace brew::template_matching
