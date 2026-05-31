#pragma once

#include <vector>

#include "brew/advanced/clustering/cluster_base.hpp"

namespace brew::clustering {

/// Connected-component labeling for measurements supplied as a 2-D image.
/// cluster()'s input Z is the intensity raster (rows = lat, cols = lon); cells
/// that are non-finite or <= 0 are background. Adjacent "on" cells (8-conn,
/// with background gaps up to closing_radius bridged) form one component,
/// emitted as columns [lon; lat; intensity] via the stored coordinate vectors.
/// Components with fewer than min_size cells are dropped. O(N) in grid cells.
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

} // namespace brew::clustering
