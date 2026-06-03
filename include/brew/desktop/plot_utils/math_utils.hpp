#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace brew::plot_utils {

double chi2inv(double p, int dof);

double normpdf(double x, double mu = 0.0, double sigma = 1.0);

std::vector<double> linspace(double start, double stop, int n);

Eigen::MatrixXd generate_ellipse_points(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const std::vector<int>& plt_inds,
    double scale = 1.0,
    int n_points = 100);

struct EllipsoidMesh {
    Eigen::MatrixXd X, Y, Z;
};

EllipsoidMesh generate_ellipsoid_mesh(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const std::vector<int>& plt_inds,
    double scale = 1.0,
    int n = 30);

EllipsoidMesh unit_sphere(int n = 30);

std::vector<double> eigen_to_vec(const Eigen::VectorXd& v);

std::vector<std::vector<double>> eigen_to_mat(const Eigen::MatrixXd& m);

}
