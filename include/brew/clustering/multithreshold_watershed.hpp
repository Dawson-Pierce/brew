#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

// @mex clustering
// @mex_name MultiThresholdWatershed
// @mex_args lon:vec, lat:vec, thresholds:vec, n_min:int, min_distance:int, min_size:int
class MultiThresholdWatershed : public ClusterBase {
public:
    MultiThresholdWatershed(std::vector<double> lon = {}, std::vector<double> lat = {},
                            std::vector<double> thresholds = {}, int n_min = 1,
                            int min_distance = 1, int min_size = 1);

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const override;

private:
    std::vector<double> lon_;
    std::vector<double> lat_;
    std::vector<double> thresholds_;
    int n_min_;
    int min_distance_;
    int min_size_;
};

}
