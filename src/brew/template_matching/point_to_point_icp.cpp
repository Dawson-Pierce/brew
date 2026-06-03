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
    Eigen::Vector3d t = params_.use_target_centroid_init
        ? Eigen::Vector3d(target.points().rowwise().mean())
        : t_init;

    IcpResult result;
    result.rotation = R;
    result.translation = t;

    for (int iter = 0; iter < params_.max_iterations; ++iter) {

        Eigen::MatrixXd src_transformed = R * source.points();
        src_transformed.colwise() += t;

        std::vector<std::pair<int,int>> corr;
        if (params_.reverse_correspondences) {
            for (auto [ti, si] : find_correspondences(
                    target.points(), src_transformed, params_.max_correspondence_dist))
                corr.emplace_back(si, ti);
        } else {
            corr = find_correspondences(
                src_transformed, target.points(), params_.max_correspondence_dist);
        }

        if (corr.empty()) break;

        const int N = static_cast<int>(corr.size());

        Eigen::Vector3d centroid_src = Eigen::Vector3d::Zero();
        Eigen::Vector3d centroid_tgt = Eigen::Vector3d::Zero();
        for (const auto& [si, ti] : corr) {
            centroid_src += src_transformed.col(si);
            centroid_tgt += target.points().col(ti);
        }
        centroid_src /= N;
        centroid_tgt /= N;

        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for (const auto& [si, ti] : corr) {
            W += (src_transformed.col(si) - centroid_src) *
                 (target.points().col(ti) - centroid_tgt).transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2, 2) = (V * U.transpose()).determinant();

        Eigen::Matrix3d delta_R = V * D * U.transpose();
        Eigen::Vector3d delta_t = centroid_tgt - delta_R * centroid_src;

        R = delta_R * R;
        t = delta_R * t + delta_t;

        double sum_sq = 0.0;
        Eigen::MatrixXd src_new = R * source.points();
        src_new.colwise() += t;
        for (const auto& [si, ti] : corr) {
            sum_sq += (src_new.col(si) - target.points().col(ti)).squaredNorm();
        }
        double rmse = std::sqrt(sum_sq / N);

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

    Eigen::MatrixXd src_final = R * source.points();
    src_final.colwise() += t;
    auto final_corr = find_correspondences(
        target.points(), src_final, params_.max_correspondence_dist);
    result.log_likelihood = compute_log_likelihood(
        target.points(), src_final, final_corr);
    result.inlier_ratio = static_cast<double>(final_corr.size()) / target.num_points();

    return result;
}

double PointToPointIcp::log_likelihood(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t) const {

    Eigen::MatrixXd src_transformed = R * source.points();
    src_transformed.colwise() += t;

    auto correspondences = find_correspondences(
        target.points(), src_transformed, params_.max_correspondence_dist);

    return compute_log_likelihood(
        target.points(), src_transformed, correspondences);
}

}
