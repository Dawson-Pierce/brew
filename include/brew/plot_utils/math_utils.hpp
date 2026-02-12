#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace brew::plot_utils {

/// Chi-squared inverse CDF (quantile function).
/// For dof=2: exact closed-form. For general dof: Wilson-Hilferty approximation.
double chi2inv(double p, int dof);

/// Standard normal PDF: (1/sqrt(2*pi*sigma^2)) * exp(-0.5*((x-mu)/sigma)^2)
double normpdf(double x, double mu = 0.0, double sigma = 1.0);

/// Generate evenly spaced values from start to stop (inclusive), matching MATLAB linspace.
std::vector<double> linspace(double start, double stop, int n);

/// Generate 2D ellipse points from center, covariance submatrix, and scaling.
/// plt_inds: 0-based indices into the state vector (must have 2 elements).
/// Returns Nx2 matrix of ellipse boundary points.
Eigen::MatrixXd generate_ellipse_points(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const std::vector<int>& plt_inds,
    double scale = 1.0,
    int n_points = 100);

/// Generate 3D ellipsoid mesh from center, covariance submatrix, and scaling.
/// plt_inds: 0-based indices into the state vector (must have 3 elements).
/// Returns {X, Y, Z} matrices each of size (n+1)x(n+1) for surf plotting.
struct EllipsoidMesh {
    Eigen::MatrixXd X, Y, Z;
};

EllipsoidMesh generate_ellipsoid_mesh(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const std::vector<int>& plt_inds,
    double scale = 1.0,
    int n = 30);

/// Generate unit sphere mesh matching MATLAB's sphere(n).
/// Returns {X, Y, Z} matrices each of size (n+1)x(n+1).
EllipsoidMesh unit_sphere(int n = 30);

/// Convert Eigen vector to std::vector<double>.
std::vector<double> eigen_to_vec(const Eigen::VectorXd& v);

/// Convert Eigen matrix to vector of vectors (row-major).
std::vector<std::vector<double>> eigen_to_mat(const Eigen::MatrixXd& m);

} // namespace brew::plot_utils

