#pragma once

#include <vector>

#include "brew/advanced/clustering/cluster_base.hpp"

namespace brew::clustering {

/// Two-scale (coarse->fine) watershed for measurements supplied as a 2-D image.
/// cluster()'s input Z is the intensity raster (rows = lat, cols = lon); cells
/// that are non-finite or <= 0 are background.
///
/// Coarse pass: local maxima thinned to a separation of region_dist cells seed a
/// watershed flood that carves the masked field into large "systems" (the gather
/// step). Fine pass: maxima thinned to min_seed_dist seed a second flood that
/// decomposes each system into cells -- but a cell is constrained to its parent
/// system (a fine basin cannot cross a coarse system boundary). Only the fine
/// cells are emitted as columns [lon; lat; intensity] (no nested objects). Cells
/// smaller than min_size or with peak intensity below min_core are dropped.
///
/// With region_dist == min_seed_dist this reduces to a single-scale watershed.
// @mex clustering
// @mex_name MultiScaleWatershed
// @mex_args lon:vec, lat:vec, region_dist:int, min_seed_dist:int, min_size:int, min_core:double
class MultiScaleWatershed : public ClusterBase {
public:
    MultiScaleWatershed(std::vector<double> lon = {}, std::vector<double> lat = {},
                        int region_dist = 40, int min_seed_dist = 10,
                        int min_size = 1, double min_core = 0.0);

    [[nodiscard]] std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const override;

private:
    std::vector<double> lon_;
    std::vector<double> lat_;
    int region_dist_;
    int min_seed_dist_;
    int min_size_;
    double min_core_;
};

} // namespace brew::clustering
