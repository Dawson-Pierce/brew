#pragma once

#include <Eigen/Dense>
#include <vector>

namespace brew::clustering {

/// DBSCAN clustering using Mahalanobis distance.
/// Mirrors MATLAB: BREW.clustering.Mahalanobis_DBSCAN
class MahalanobisDBSCAN {
public:
    MahalanobisDBSCAN(double epsilon = 1.0, int min_pts = 3);

    /// Cluster columns of Z using Mahalanobis distance with covariance S.
    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z,
        const Eigen::MatrixXd& S) const;

    [[nodiscard]] std::vector<Eigen::VectorXd> get_unclustered(
        const Eigen::MatrixXd& Z,
        const Eigen::MatrixXd& S) const;

private:
    double epsilon_;
    int min_pts_;

    [[nodiscard]] std::vector<int> run_dbscan(
        const Eigen::MatrixXd& Z,
        const Eigen::MatrixXd& S_inv) const;
};

} // namespace brew::clustering
