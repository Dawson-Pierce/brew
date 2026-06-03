#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

// @mex clustering
// @mex_name GridCC
// @mex_args lon:vec, lat:vec, closing_radius:int, min_size:int
class GridCC : public ClusterBase {
public:
    GridCC(std::vector<double> lon = {}, std::vector<double> lat = {},
           int closing_radius = 0, int min_size = 1);

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const override;

private:
    std::vector<double> lon_;
    std::vector<double> lat_;
    int closing_radius_;
    int min_size_;
};

}
