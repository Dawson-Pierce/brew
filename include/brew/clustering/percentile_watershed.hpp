#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

// @mex clustering
// @mex_name PercentileWatershed
// @mex_args lon:vec, lat:vec, percentile:double, n_min:int, min_distance:int, min_size:int, min_core_abs:double
class PercentileWatershed : public ClusterBase {
public:
    PercentileWatershed(std::vector<double> lon = {}, std::vector<double> lat = {},
                        double percentile = 95.0, int n_min = 1,
                        int min_distance = 1, int min_size = 1,
                        double min_core_abs = 0.0);

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const override;

private:
    std::vector<double> lon_;
    std::vector<double> lat_;
    double percentile_;
    int n_min_;
    int min_distance_;
    int min_size_;
    double min_core_abs_;
};

}
