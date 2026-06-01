#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

/// Connected-components-then-watershed for measurements supplied as a 2-D
/// image. cluster()'s input Z is the intensity raster (rows = lat, cols = lon);
/// cells that are non-finite or <= 0 are background.
///
/// Stage 1 (regions): the on-mask is dilated by closing_radius cells (Chebyshev)
/// and connected-component-labelled, so on-cell blobs separated by sub-threshold
/// gaps of up to 2*closing_radius are gathered into a single region. With
/// closing_radius = 0 this reduces to plain connected components on the mask.
///
/// Stage 2 (cells within region): local maxima of Z are thinned to a separation
/// of min_seed_dist cells; each surviving maximum seeds a priority flood that
/// is constrained to its own region (a fine basin cannot cross a region
/// boundary). All seeded basins in a region are emitted, so a region with
/// multiple hot spots splits into multiple cells. Regions with NO surviving
/// seed fall back to a single basin covering all their on-cells; seeded
/// regions whose flood cannot reach every on-cell (because the closing bridge
/// runs through off-cells) leave the unreachable cells dropped rather than
/// emitting them as a separate "surrounding" basin.
///
/// Emitted as columns [lon; lat; intensity]. Cells smaller than min_size or
/// with peak intensity below min_core are dropped.
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

} // namespace brew::clustering
