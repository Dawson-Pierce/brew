#include "brew/plot_utils/math_utils.hpp"
#include <stdexcept>
#include <numbers>

namespace brew::plot_utils {

namespace {

/// Rational approximation for the probit (inverse normal CDF) function.
/// Abramowitz & Stegun 26.2.23 — accurate to ~4.5e-4.
double probit(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::domain_error("probit: p must be in (0, 1)");
    }
    // Work in the lower tail
    const bool lower = (p <= 0.5);
    const double t_p = lower ? p : (1.0 - p);
    const double t = std::sqrt(-2.0 * std::log(t_p));

    // Rational approximation coefficients
    constexpr double c0 = 2.515517;
    constexpr double c1 = 0.802853;
    constexpr double c2 = 0.010328;
    constexpr double d1 = 1.432788;
    constexpr double d2 = 0.189269;
    constexpr double d3 = 0.001308;

    double z = t - (c0 + c1 * t + c2 * t * t) /
                       (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    return lower ? -z : z;
}

} // anonymous namespace

double chi2inv(double p, int dof) {
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    if (dof <= 0) throw std::domain_error("chi2inv: dof must be positive");

    // Exact formula for dof == 2
    if (dof == 2) {
        return -2.0 * std::log(1.0 - p);
    }

    // Wilson-Hilferty approximation for general dof
    const double d = static_cast<double>(dof);
    const double z = probit(p);
    const double h = 2.0 / (9.0 * d);
    double x = d * std::pow(1.0 - h + z * std::sqrt(h), 3.0);
    return std::max(0.0, x);
}

double normpdf(double x, double mu, double sigma) {
    constexpr double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * std::numbers::pi);
    const double z = (x - mu) / sigma;
    return (inv_sqrt_2pi / sigma) * std::exp(-0.5 * z * z);
}

std::vector<double> linspace(double start, double stop, int n) {
    std::vector<double> result(n);
    if (n == 1) {
        result[0] = stop;
        return result;
    }
    const double step = (stop - start) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

Eigen::MatrixXd generate_ellipse_points(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const std::vector<int>& plt_inds,
    double scale,
    int n_points) {

    if (plt_inds.size() != 2) {
        throw std::invalid_argument("generate_ellipse_points: plt_inds must have 2 elements");
    }

    // Extract 2x2 sub-covariance
    Eigen::Matrix2d S;
    S(0, 0) = covariance(plt_inds[0], plt_inds[0]);
    S(0, 1) = covariance(plt_inds[0], plt_inds[1]);
    S(1, 0) = covariance(plt_inds[1], plt_inds[0]);
    S(1, 1) = covariance(plt_inds[1], plt_inds[1]);

    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(S);
    Eigen::Vector2d eigenvalues = eig.eigenvalues().cwiseMax(0.0);
    Eigen::Matrix2d eigenvectors = eig.eigenvectors();

    // Scale by sqrt(eigenvalues) * scale
    Eigen::Matrix2d L = eigenvectors *
        Eigen::Vector2d(scale * std::sqrt(eigenvalues(0)),
                        scale * std::sqrt(eigenvalues(1))).asDiagonal();

    // Generate unit circle
    auto theta = linspace(0.0, 2.0 * std::numbers::pi, n_points);

    Eigen::MatrixXd pts(n_points, 2);
    for (int i = 0; i < n_points; ++i) {
        Eigen::Vector2d unit_pt(std::cos(theta[i]), std::sin(theta[i]));
        Eigen::Vector2d pt = L * unit_pt;
        pts(i, 0) = pt(0) + mean(plt_inds[0]);
        pts(i, 1) = pt(1) + mean(plt_inds[1]);
    }
    return pts;
}

EllipsoidMesh unit_sphere(int n) {
    EllipsoidMesh mesh;
    const int rows = n + 1;
    const int cols = n + 1;
    mesh.X.resize(rows, cols);
    mesh.Y.resize(rows, cols);
    mesh.Z.resize(rows, cols);

    auto phi_vals = linspace(-std::numbers::pi / 2.0, std::numbers::pi / 2.0, rows);
    auto theta_vals = linspace(0.0, 2.0 * std::numbers::pi, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mesh.X(i, j) = std::cos(phi_vals[i]) * std::cos(theta_vals[j]);
            mesh.Y(i, j) = std::cos(phi_vals[i]) * std::sin(theta_vals[j]);
            mesh.Z(i, j) = std::sin(phi_vals[i]);
        }
    }
    return mesh;
}

EllipsoidMesh generate_ellipsoid_mesh(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const std::vector<int>& plt_inds,
    double scale,
    int n) {

    if (plt_inds.size() != 3) {
        throw std::invalid_argument("generate_ellipsoid_mesh: plt_inds must have 3 elements");
    }

    // Extract 3x3 sub-covariance
    Eigen::Matrix3d S;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S(i, j) = covariance(plt_inds[i], plt_inds[j]);
        }
    }

    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(S);
    Eigen::Vector3d eigenvalues = eig.eigenvalues().cwiseMax(0.0);
    Eigen::Matrix3d eigenvectors = eig.eigenvectors();

    // Scale by sqrt(eigenvalues) * scale
    Eigen::Matrix3d L = eigenvectors *
        Eigen::Vector3d(scale * std::sqrt(eigenvalues(0)),
                        scale * std::sqrt(eigenvalues(1)),
                        scale * std::sqrt(eigenvalues(2))).asDiagonal();

    // Generate unit sphere
    auto sphere = unit_sphere(n);
    const int rows = n + 1;
    const int cols = n + 1;

    EllipsoidMesh mesh;
    mesh.X.resize(rows, cols);
    mesh.Y.resize(rows, cols);
    mesh.Z.resize(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Eigen::Vector3d unit_pt(sphere.X(i, j), sphere.Y(i, j), sphere.Z(i, j));
            Eigen::Vector3d pt = L * unit_pt;
            mesh.X(i, j) = pt(0) + mean(plt_inds[0]);
            mesh.Y(i, j) = pt(1) + mean(plt_inds[1]);
            mesh.Z(i, j) = pt(2) + mean(plt_inds[2]);
        }
    }
    return mesh;
}

std::vector<double> eigen_to_vec(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

std::vector<std::vector<double>> eigen_to_mat(const Eigen::MatrixXd& m) {
    std::vector<std::vector<double>> result(m.rows());
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
        result[i].resize(m.cols());
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
            result[i][j] = m(i, j);
        }
    }
    return result;
}

} // namespace brew::plot_utils

