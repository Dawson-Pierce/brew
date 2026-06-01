#pragma once

#include <Eigen/Dense>
#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

/// DBSCAN clustering algorithm.
// @mex clustering
// @mex_name DBSCAN
// @mex_args epsilon:double, min_pts:int, dims:ivec, scales:vec

class DBSCAN : public ClusterBase {
public:
    /// dims selects which rows of Z participate in the neighbor distance
    /// (e.g. position rows, excluding a trailing weight). Empty = all rows.
    /// scales is a per-selected-dimension multiplier so the neighbor distance
    /// becomes sqrt(sum_d scales[d] * (z_i,d - z_j,d)^2) -- a weighted Euclidean
    /// rather than a plain geometric distance. Empty = all ones (unweighted).
    DBSCAN(double epsilon = 1.0, int min_pts = 3,
           std::vector<int> dims = {}, std::vector<double> scales = {});

    /// Cluster columns of Z (each column is a data point). Distance uses only
    /// the rows selected by dims; the returned clusters keep all rows.
    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(const Eigen::MatrixXd& Z) const override;

    /// Get unclustered (noise) points as individual column vectors.
    [[nodiscard]] std::vector<Eigen::VectorXd> get_unclustered(const Eigen::MatrixXd& Z) const;

private:
    double epsilon_;
    int min_pts_;
    std::vector<int> dims_;
    std::vector<double> scales_;

    /// Run DBSCAN and return label per point (-1 = noise, >=0 = cluster id).
    [[nodiscard]] std::vector<int> run_dbscan(const Eigen::MatrixXd& Z) const;
};

} // namespace brew::clustering
