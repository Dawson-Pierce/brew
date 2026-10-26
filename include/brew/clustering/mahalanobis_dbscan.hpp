#pragma once

#include <Eigen/Dense>
#include <vector>

namespace brew::clustering {

// @mex clustering
// @mex_name MahalanobisDBSCAN
// @mex_args epsilon:double, min_pts:int

class MahalanobisDBSCAN {
public:
    MahalanobisDBSCAN(double epsilon = 1.0, int min_pts = 3);

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

}
