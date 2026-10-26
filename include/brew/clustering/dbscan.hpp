#pragma once

#include <Eigen/Dense>
#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

// @mex clustering
// @mex_name DBSCAN
// @mex_args epsilon:double, min_pts:int, dims:ivec, scales:vec

class DBSCAN : public ClusterBase {
public:

    DBSCAN(double epsilon = 1.0, int min_pts = 3,
           std::vector<int> dims = {}, std::vector<double> scales = {});

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(const Eigen::MatrixXd& Z) const override;

    [[nodiscard]] std::vector<Eigen::VectorXd> get_unclustered(const Eigen::MatrixXd& Z) const;

private:
    double epsilon_;
    int min_pts_;
    std::vector<int> dims_;
    std::vector<double> scales_;

    [[nodiscard]] std::vector<int> run_dbscan(const Eigen::MatrixXd& Z) const;
};

}
