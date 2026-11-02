#pragma once

#include <Eigen/Dense>
#include <vector>

#include "brew/utils/clustering/cluster_base.hpp"

namespace brew::utils::clustering {

// @mex clustering
// @mex_name DBSCAN
// @mex_args epsilon:double, min_pts:int, dims:ivec, scales:vec

class DBSCAN : public ClusterBase {
public:

    DBSCAN(double epsilon = 1.0, int min_pts = 3,
           std::vector<int> dims = {}, std::vector<double> scales = {});

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(const Eigen::MatrixXd& Z) const override;

    [[nodiscard]] std::vector<Eigen::VectorXd> get_unclustered(const Eigen::MatrixXd& Z) const;

    // Cluster membership in input order; -1 denotes noise.
    [[nodiscard]] std::vector<int> labels(const Eigen::MatrixXd& Z) const { return run_dbscan(Z); }

private:
    double epsilon_;
    int min_pts_;
    std::vector<int> dims_;
    std::vector<double> scales_;

    [[nodiscard]] std::vector<int> run_dbscan(const Eigen::MatrixXd& Z) const;
};

}
