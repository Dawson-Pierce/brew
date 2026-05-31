#pragma once

#include <vector>

#include "brew/advanced/clustering/cluster_base.hpp"

namespace brew::clustering {

/// Marker-controlled watershed segmentation for measurements supplied as a 2-D
/// image. cluster()'s input Z is the intensity raster (rows = lat, cols = lon);
/// cells that are non-finite or <= 0 are background.
///
/// Storm cores are the local maxima of the field; maxima are thinned so that no
/// two seeds are within min_seed_dist cells (the higher wins), which sets the
/// minimum core separation for two cells to count as distinct storms. The masked
/// field is then flooded from the seeds in order of decreasing intensity, so two
/// touching storms split at the intensity valley between their cores -- unlike
/// connected components, which would fuse them. Each basin is emitted as columns
/// [lon; lat; intensity]; basins smaller than min_size cells, or whose peak
/// intensity is below min_core, are dropped (min_core = 0 keeps all).
/// Near-linear: O(N) maxima + O(N log N) flood. O(N) in grid cells.
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

} // namespace brew::clustering
