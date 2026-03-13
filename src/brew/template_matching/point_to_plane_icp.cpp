#include "brew/template_matching/point_to_plane_icp.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace brew::template_matching {

std::unique_ptr<IcpBase> PointToPlaneIcp::clone() const {
    auto c = std::make_unique<PointToPlaneIcp>();
    c->params_ = params_;
    return c;
}

Eigen::MatrixXd PointToPlaneIcp::estimate_normals(const PointCloud& cloud, int k) {
    const int N = cloud.num_points();
    const int d = cloud.dim();
    Eigen::MatrixXd normals(d, N);

    k = std::min(k, N);

    for (int i = 0; i < N; ++i) {
        // Find k nearest neighbors by brute force
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&](int a, int b) {
                return (cloud.points().col(a) - cloud.points().col(i)).squaredNorm()
                     < (cloud.points().col(b) - cloud.points().col(i)).squaredNorm();
            });

        // Local PCA: build covariance of k neighbors
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int j = 0; j < k; ++j) {
            centroid += cloud.points().col(indices[j]);
        }
        centroid /= k;

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int j = 0; j < k; ++j) {
            Eigen::Vector3d diff = cloud.points().col(indices[j]) - centroid;
            cov += diff * diff.transpose();
        }
        cov /= k;

        // Normal is eigenvector of smallest eigenvalue
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        normals.col(i) = solver.eigenvectors().col(0);
    }

    return normals;
}

IcpResult PointToPlaneIcp::align(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& t_init) const {

    // Estimate target normals
    const int k = std::min(10, target.num_points());
    Eigen::MatrixXd normals = estimate_normals(target, k);

    Eigen::Matrix3d R = R_init;
    Eigen::Vector3d t = t_init;

    IcpResult result;
    result.rotation = R;
    result.translation = t;

    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        // Transform source
        Eigen::MatrixXd src_transformed = R * source.points();
        src_transformed.colwise() += t;

        // Find correspondences
        auto correspondences = find_correspondences(
            src_transformed, target.points(), params_.max_correspondence_dist);

        if (correspondences.empty()) break;

        const int N = static_cast<int>(correspondences.size());

        // Build 6×6 linear system for point-to-plane
        // Minimize Σ (n_i · (R*s_i + t - t_i))²
        // Linearize R ≈ I + [α]× for small angles
        // Variables: x = [α, β, γ, tx, ty, tz]
        Eigen::MatrixXd A(N, 6);
        Eigen::VectorXd b(N);

        for (int idx = 0; idx < N; ++idx) {
            const auto& [si, ti] = correspondences[idx];
            const Eigen::Vector3d& p = src_transformed.col(si);
            const Eigen::Vector3d& q = target.points().col(ti);
            const Eigen::Vector3d& n = normals.col(ti);

            // A row: [p × n, n]
            A(idx, 0) = p(1) * n(2) - p(2) * n(1);
            A(idx, 1) = p(2) * n(0) - p(0) * n(2);
            A(idx, 2) = p(0) * n(1) - p(1) * n(0);
            A(idx, 3) = n(0);
            A(idx, 4) = n(1);
            A(idx, 5) = n(2);

            b(idx) = n.dot(q - p);
        }

        // Solve A^T A x = A^T b
        Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);

        // Extract incremental rotation (small angle approximation)
        Eigen::Matrix3d delta_R;
        delta_R << 1.0,   -x(2),  x(1),
                   x(2),   1.0,  -x(0),
                  -x(1),   x(0),  1.0;

        // Re-orthogonalize via SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(delta_R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        delta_R = svd.matrixU() * svd.matrixV().transpose();

        Eigen::Vector3d delta_t = x.tail<3>();

        // Update cumulative transform
        R = delta_R * R;
        t = delta_R * t + delta_t;

        // Compute RMSE (point-to-plane)
        double sum_sq = 0.0;
        Eigen::MatrixXd src_new = R * source.points();
        src_new.colwise() += t;
        for (const auto& [si, ti] : correspondences) {
            double d = normals.col(ti).dot(src_new.col(si) - target.points().col(ti));
            sum_sq += d * d;
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

double PointToPlaneIcp::likelihood(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t) const {

    // Use point-to-plane distances for likelihood
    const int k = std::min(10, target.num_points());
    Eigen::MatrixXd normals = estimate_normals(target, k);

    Eigen::MatrixXd src_transformed = R * source.points();
    src_transformed.colwise() += t;

    auto correspondences = find_correspondences(
        src_transformed, target.points(), params_.max_correspondence_dist);

    if (correspondences.empty()) return 0.0;

    const int N = static_cast<int>(correspondences.size());

    // Point-to-plane residuals
    double sum_sq = 0.0;
    for (const auto& [si, ti] : correspondences) {
        double d = normals.col(ti).dot(
            src_transformed.col(si) - target.points().col(ti));
        sum_sq += d * d;
    }

    // 1D Gaussian per residual (scalar plane distance)
    double log_L = -0.5 * N * std::log(2.0 * M_PI * params_.sigma_sq)
                    - sum_sq / (2.0 * params_.sigma_sq);

    return std::exp(log_L);
}

} // namespace brew::template_matching
