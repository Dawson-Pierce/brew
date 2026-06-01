#pragma once

#include <vector>

#include "brew/clustering/cluster_base.hpp"

namespace brew::clustering {

/// tobac-style multi-threshold feature detection + watershed segmentation for
/// measurements supplied as a 2-D image. cluster()'s input Z is the intensity
/// raster (rows = lat, cols = lon); cells that are non-finite or <= 0 are
/// background, and the on-mask is the full segmentation domain (its outer edge
/// sets storm extent).
///
/// Feature detection: at each level in thresholds (ascending) the on-cells with
/// Z >= level are connected-component labelled, and every component of at least
/// n_min cells contributes its peak as a candidate core. A broad low-level blob
/// that breaks into several high-level cores therefore yields one feature per
/// core. Candidates are kept strongest-first and any within min_distance cells
/// of an already-kept feature are discarded (this also dedupes the same core
/// seen across levels). Segmentation: a priority flood from the surviving
/// features partitions the whole on-mask into cells; on-regions that contain no
/// feature are dropped, and cells smaller than min_size cells are dropped. Each
/// cell is emitted as columns [lon; lat; intensity].
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

} // namespace brew::clustering
