#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

// @mex clustering
// @mex_name Watershed
// @mex_args lon:vec, lat:vec, min_seed_dist:int, min_size:int, min_core:double
class Watershed : public ClusterBase {
public:
    Watershed(std::vector<double> lon = {}, std::vector<double> lat = {},
              int min_seed_dist = 3, int min_size = 1, double min_core = 0.0);

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const override;

private:
    std::vector<double> lon_;
    std::vector<double> lat_;
    int min_seed_dist_;
    int min_size_;
    double min_core_;
};

}
