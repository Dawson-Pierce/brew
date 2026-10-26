#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

// @mex clustering
// @mex_name CCWatershed
// @mex_args lon:vec, lat:vec, closing_radius:int, min_seed_dist:int, min_size:int, min_core:double
class CCWatershed : public ClusterBase {
public:
    CCWatershed(std::vector<double> lon = {}, std::vector<double> lat = {},
                int closing_radius = 0, int min_seed_dist = 1,
                int min_size = 1, double min_core = 0.0);

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const override;

private:
    std::vector<double> lon_;
    std::vector<double> lat_;
    int closing_radius_;
    int min_seed_dist_;
    int min_size_;
    double min_core_;
};

}
