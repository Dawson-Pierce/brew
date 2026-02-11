#pragma once

#include <Eigen/Dense>
#include <vector>

namespace brew::clustering {

/// DBSCAN clustering algorithm.
/// Mirrors MATLAB: BREW.clustering.DBSCAN_obj
class DBSCAN {
public:
    DBSCAN(double epsilon = 1.0, int min_pts = 3);

    /// Cluster columns of Z (each column is a data point).
    /// Returns a vector of clusters, where each cluster is a matrix
    /// whose columns are the points belonging to that cluster.
    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(const Eigen::MatrixXd& Z) const;

    /// Get unclustered (noise) points as individual column vectors.
    [[nodiscard]] std::vector<Eigen::VectorXd> get_unclustered(const Eigen::MatrixXd& Z) const;

private:
    double epsilon_;
    int min_pts_;

    /// Run DBSCAN and return label per point (-1 = noise, >=0 = cluster id).
    [[nodiscard]] std::vector<int> run_dbscan(const Eigen::MatrixXd& Z) const;
};

} // namespace brew::clustering
