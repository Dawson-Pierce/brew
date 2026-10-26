#include "brew/template_matching/pca_icp.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace brew::template_matching {

PcaIcp::PcaIcp(std::unique_ptr<IcpBase> inner_icp, const PointCloud& source_template)
    : inner_(std::move(inner_icp)) {
    if (!inner_) throw std::invalid_argument("PcaIcp requires an inner ICP implementation.");
    params_ = inner_->params();
    src_axes_ = pca_axes(source_template.points());
    src_centroid_ = source_template.points().rowwise().mean();
}

Eigen::Matrix3d PcaIcp::pca_axes(const Eigen::MatrixXd& points) {
    if (points.rows() != 3 || points.cols() < 2) {
        throw std::invalid_argument("PCA initialization requires a 3xN cloud with N at least two.");
    }
    Eigen::Vector3d mean = points.rowwise().mean();
    Eigen::MatrixXd centered = points.colwise() - mean;
    Eigen::Matrix3d cov = (centered * centered.transpose()) / (points.cols() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d axes;
    axes.col(0) = solver.eigenvectors().col(2);
    axes.col(1) = solver.eigenvectors().col(1);
    axes.col(2) = solver.eigenvectors().col(0);

    if (axes.determinant() < 0) {
        axes.col(2) = -axes.col(2);
    }
    return axes;
}

std::vector<Eigen::Matrix3d> PcaIcp::pca_candidates(
    const Eigen::Matrix3d& src_axes, const Eigen::Matrix3d& tgt_axes,
    const Eigen::Matrix3d& predicted) {

    std::vector<Eigen::Matrix3d> candidates{predicted};
    std::array<int, 3> permutation{0, 1, 2};
    do {
        for (int s0 : {-1, 1}) {
            for (int s1 : {-1, 1}) {
                for (int s2 : {-1, 1}) {
                    Eigen::Matrix3d oriented;
                    oriented.col(0) = s0 * src_axes.col(permutation[0]);
                    oriented.col(1) = s1 * src_axes.col(permutation[1]);
                    oriented.col(2) = s2 * src_axes.col(permutation[2]);
                    if (oriented.determinant() > 0.0) {
                        candidates.push_back(tgt_axes * oriented.transpose());
                    }
                }
            }
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return candidates;
}

IcpResult PcaIcp::align(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& ) const {

    Eigen::Matrix3d tgt_axes = pca_axes(target.points());
    Eigen::Vector3d tgt_centroid = target.points().rowwise().mean();

    auto candidates = pca_candidates(src_axes_, tgt_axes, R_init);

    const int step = std::max(1, source.num_points() / 100);
    std::vector<double> costs(candidates.size());
    for (std::size_t c = 0; c < candidates.size(); ++c) {
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

    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return costs[a] < costs[b]; });

    auto candidate_icp = inner_->clone();
    candidate_icp->set_params(params_);

    const int n_tries = std::min<int>(3, order.size());
    IcpResult best_result;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_tries; ++i) {
        Eigen::Matrix3d R_cand = candidates[order[i]];
        Eigen::Vector3d t_cand = tgt_centroid - R_cand * src_centroid_;
        auto result = candidate_icp->align(source, target, R_cand, t_cand);
        double cost = 0.0;
        for (int j = 0; j < target.num_points(); ++j) {
            const Eigen::Vector3d local = result.rotation.transpose()
                * (target.points().col(j) - result.translation);
            cost += source.nearest(local).second;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_result = result;
        }
    }

    return best_result;
}

double PcaIcp::log_likelihood(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t) const {
    auto configured = inner_->clone();
    configured->set_params(params_);
    return configured->log_likelihood(source, target, R, t);
}

std::unique_ptr<IcpBase> PcaIcp::clone() const {
    auto cloned = std::unique_ptr<PcaIcp>(new PcaIcp());
    cloned->inner_ = inner_->clone();
    cloned->src_axes_ = src_axes_;
    cloned->src_centroid_ = src_centroid_;
    cloned->params_ = params_;
    return cloned;
}

}
