#include "brew/template_matching/point_to_plane_icp.hpp"
#include "brew/template_matching/so3.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace brew::template_matching {

std::unique_ptr<IcpBase> PointToPlaneIcp::clone() const {
    auto c = std::make_unique<PointToPlaneIcp>();
    c->params_ = params_;
    return c;
}

Eigen::MatrixXd PointToPlaneIcp::estimate_normals(const PointCloud& cloud, int k) {
    return cloud.normals(k);
}

IcpResult PointToPlaneIcp::align(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& t_init) const {

    const int k = std::min(10, source.num_points());
    Eigen::MatrixXd src_normals = estimate_normals(source, k);

    Eigen::Matrix3d R = R_init;
    Eigen::Vector3d t = params_.use_target_centroid_init
        ? Eigen::Vector3d(target.points().rowwise().mean()
                          - R * source.points().rowwise().mean())
        : t_init;
    const Eigen::Vector3d target_center = target.points().rowwise().mean();

    IcpResult result;
    result.rotation = R;
    result.translation = t;
    const int query_count = params_.reverse_correspondences
        ? target.num_points() : source.num_points();
    double best_fitness = std::numeric_limits<double>::infinity();
    Eigen::Matrix3d best_R = R;
    Eigen::Vector3d best_t = t;
    auto retain_best = [&](const Eigen::MatrixXd& transformed,
                           const std::vector<std::pair<int, int>>& correspondences) {
        // A plane residual alone cannot detect sliding along a face. Retain
        // the best finite-surface fit visited by the point-to-plane iterations.
        double fitness = (query_count - static_cast<int>(correspondences.size()))
            * params_.max_correspondence_dist * params_.max_correspondence_dist;
        for (const auto& [si, ti] : correspondences) {
            fitness += (transformed.col(si) - target.points().col(ti)).squaredNorm();
        }
        if (!correspondences.empty() && fitness < best_fitness) {
            best_fitness = fitness;
            best_R = R;
            best_t = t;
        }
    };

    for (int iter = 0; iter < params_.max_iterations; ++iter) {

        Eigen::MatrixXd src_transformed = R * source.points();
        src_transformed.colwise() += t;
        Eigen::MatrixXd normals_transformed = R * src_normals;

        std::vector<std::pair<int,int>> corr;
        if (params_.reverse_correspondences) {
            corr = find_reverse_correspondences(
                source, target.points(), R, t, params_.max_correspondence_dist);
        } else {
            corr = find_correspondences(
                src_transformed, target.points(), params_.max_correspondence_dist);
        }

        if (corr.empty()) break;
        retain_best(src_transformed, corr);

        const int N = static_cast<int>(corr.size());

        Eigen::MatrixXd A(N, 6);
        Eigen::VectorXd b(N);

        for (int idx = 0; idx < N; ++idx) {
            const auto& [si, ti] = corr[idx];
            const Eigen::Vector3d p = source.points().col(si);
            const Eigen::Vector3d q = R.transpose() * (target.points().col(ti) - t);
            const Eigen::Vector3d n = src_normals.col(si);

            // Fit observed returns in the fixed template frame. Its normals
            // then stay fixed during the conventional point-to-plane step.
            const Eigen::Vector3d cross = q.cross(n);
            A(idx, 0) = cross(0);
            A(idx, 1) = cross(1);
            A(idx, 2) = cross(2);
            A(idx, 3) = n(0);
            A(idx, 4) = n(1);
            A(idx, 5) = n(2);

            b(idx) = n.dot(p - q);
        }

        // Express rotational and translational columns on the same length
        // scale before identifying directions unsupported by the visible patch.
        Eigen::VectorXd x;
        if (params_.observability_threshold > 0.0) {
            const Eigen::MatrixXd centered = source.points().colwise()
                - source.points().rowwise().mean();
            const double radius = std::max(1e-9,
                std::sqrt(centered.squaredNorm() / source.num_points()));
            A.leftCols(3) /= radius;
            Eigen::JacobiSVD<Eigen::MatrixXd> solver(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const double relative_residual = std::sqrt(b.squaredNorm() / N) / radius;
            solver.setThreshold(std::min(params_.observability_threshold,
                                         std::max(1e-6, 10.0 * relative_residual)));
            x = solver.solve(b);
            x.head<3>() /= radius;
        } else {
            x = A.colPivHouseholderQr().solve(b);
        }

        double initial_sq = 0.0;
        for (const auto& [si, ti] : corr) {
            const double d = normals_transformed.col(si).dot(
                src_transformed.col(si) - target.points().col(ti));
            initial_sq += d * d;
        }

        double sum_sq = initial_sq;
        bool accepted = false;
        const Eigen::Matrix3d inverse_R = R.transpose();
        const Eigen::Vector3d inverse_t = inverse_R * (target_center - t);
        for (double scale = 1.0; scale >= 1.0 / 4096.0; scale *= 0.5) {
            const Eigen::Matrix3d delta_R = so3::exp(scale * Eigen::Vector3d(x.head<3>()));
            const Eigen::Matrix3d trial_R = (delta_R * inverse_R).transpose();
            const Eigen::Vector3d trial_inverse_t = delta_R * inverse_t + scale * x.tail<3>();
            const Eigen::Vector3d trial_t = target_center - trial_R * trial_inverse_t;
            double trial_sq = 0.0;
            for (const auto& [si, ti] : corr) {
                const Eigen::Vector3d p = trial_R * source.points().col(si) + trial_t;
                const Eigen::Vector3d n = trial_R * src_normals.col(si);
                const double d = n.dot(p - target.points().col(ti));
                trial_sq += d * d;
            }
            if (trial_sq < initial_sq) {
                R = trial_R;
                t = trial_t;
                sum_sq = trial_sq;
                accepted = true;
                break;
            }
        }
        if (!accepted) break;

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
    std::vector<std::pair<int, int>> final_corr;
    if (params_.reverse_correspondences) {
        final_corr = find_reverse_correspondences(
            source, target.points(), R, t, params_.max_correspondence_dist);
    } else {
        final_corr = find_correspondences(
            src_final, target.points(), params_.max_correspondence_dist);
    }
    retain_best(src_final, final_corr);
    result.rotation = R = best_R;
    result.translation = t = best_t;
    src_final = R * source.points();
    src_final.colwise() += t;
    final_corr = params_.reverse_correspondences
        ? find_reverse_correspondences(source, target.points(), R, t,
                                       params_.max_correspondence_dist)
        : find_correspondences(src_final, target.points(), params_.max_correspondence_dist);
    double plane_sum_sq = 0.0;
    for (const auto& [si, ti] : final_corr) {
        const double residual = (R * src_normals.col(si)).dot(
            src_final.col(si) - target.points().col(ti));
        plane_sum_sq += residual * residual;
    }
    result.rmse = final_corr.empty() ? std::numeric_limits<double>::infinity()
        : std::sqrt(plane_sum_sq / static_cast<double>(final_corr.size()));
    result.log_likelihood = compute_log_likelihood(
        src_final, target.points(), final_corr);
    result.inlier_ratio = query_count > 0
        ? static_cast<double>(final_corr.size()) / query_count : 0.0;

    return result;
}

double PointToPlaneIcp::log_likelihood(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t) const {

    Eigen::MatrixXd src_transformed = R * source.points();
    src_transformed.colwise() += t;

    std::vector<std::pair<int, int>> correspondences;
    if (params_.reverse_correspondences) {
        correspondences = find_reverse_correspondences(
            source, target.points(), R, t, params_.max_correspondence_dist);
    } else {
        correspondences = find_correspondences(
            src_transformed, target.points(), params_.max_correspondence_dist);
    }

    return compute_log_likelihood(
        src_transformed, target.points(), correspondences);
}

}
