#include "brew/template_matching/point_to_point_icp.hpp"
#include <cmath>

namespace brew::template_matching {

std::unique_ptr<IcpBase> PointToPointIcp::clone() const {
    auto c = std::make_unique<PointToPointIcp>();
    c->params_ = params_;
    return c;
}

IcpResult PointToPointIcp::align(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& t_init) const {

    Eigen::Matrix3d R = R_init;
    Eigen::Vector3d t = t_init;

    IcpResult result;
    result.rotation = R;
    result.translation = t;

    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        // Transform source points
        Eigen::MatrixXd src_transformed = R * source.points();
        src_transformed.colwise() += t;

        // Find correspondences
        auto correspondences = find_correspondences(
            src_transformed, target.points(), params_.max_correspondence_dist);

        if (correspondences.empty()) break;

        const int N = static_cast<int>(correspondences.size());

        // Compute centroids of matched points
        Eigen::Vector3d centroid_src = Eigen::Vector3d::Zero();
        Eigen::Vector3d centroid_tgt = Eigen::Vector3d::Zero();
        for (const auto& [si, ti] : correspondences) {
            centroid_src += src_transformed.col(si);
            centroid_tgt += target.points().col(ti);
        }
        centroid_src /= N;
        centroid_tgt /= N;

        // Build cross-covariance matrix
        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for (const auto& [si, ti] : correspondences) {
            W += (src_transformed.col(si) - centroid_src) *
                 (target.points().col(ti) - centroid_tgt).transpose();
        }

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        // Handle reflection
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2, 2) = (V * U.transpose()).determinant();

        Eigen::Matrix3d delta_R = V * D * U.transpose();
        Eigen::Vector3d delta_t = centroid_tgt - delta_R * centroid_src;

        // Update cumulative transform
        R = delta_R * R;
        t = delta_R * t + delta_t;

        // Compute RMSE
        double sum_sq = 0.0;
        Eigen::MatrixXd src_new = R * source.points();
        src_new.colwise() += t;
        for (const auto& [si, ti] : correspondences) {
            sum_sq += (src_new.col(si) - target.points().col(ti)).squaredNorm();
        }
        double rmse = std::sqrt(sum_sq / N);

        // Check convergence
        double delta_rmse = std::abs(rmse - result.rmse);
        result.rotation = R;
        result.translation = t;
        result.rmse = rmse;
        result.iterations = iter + 1;

        if (delta_rmse < params_.tolerance) {
            result.converged = true;
            break;
        }
    }

    // Compute final likelihood and inlier ratio
    Eigen::MatrixXd src_final = R * source.points();
    src_final.colwise() += t;
    auto final_corr = find_correspondences(
        src_final, target.points(), params_.max_correspondence_dist);
    result.likelihood = compute_gaussian_likelihood(
        src_final, target.points(), final_corr);
    result.inlier_ratio = static_cast<double>(final_corr.size()) / source.num_points();

    return result;
}

double PointToPointIcp::likelihood(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t) const {

    Eigen::MatrixXd src_transformed = R * source.points();
    src_transformed.colwise() += t;

    auto correspondences = find_correspondences(
        src_transformed, target.points(), params_.max_correspondence_dist);

    return compute_gaussian_likelihood(
        src_transformed, target.points(), correspondences);
}

} // namespace brew::template_matching
