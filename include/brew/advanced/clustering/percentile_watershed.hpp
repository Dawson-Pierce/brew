#pragma once

#include <vector>

#include "brew/advanced/clustering/cluster_base.hpp"

namespace brew::clustering {

/// Adaptive-percentile watershed for measurements supplied as a 2-D image.
/// cluster()'s input Z is the intensity raster (rows = lat, cols = lon); cells
/// that are non-finite or <= 0 are background, and the on-mask is the full
/// segmentation domain (its outer edge sets storm extent).
///
/// For each connected region of on-cells the "core" is defined as cells whose
/// value is at least the in-region `percentile` quantile AND at least
/// min_core_abs. A region's core may break into multiple sub-components; each
/// sub-component of at least n_min cells contributes its peak as a candidate
/// seed. Candidates are kept strongest-first and any within min_distance cells
/// of an already-kept seed are dropped. A priority flood from the surviving
/// seeds partitions the whole on-mask into cells; regions that produce no
/// qualifying core are dropped, as are cells smaller than min_size cells.
/// Each cell is emitted as columns [lon; lat; intensity].
///
/// percentile is in [0, 100] (e.g. 95 picks the top 5% of each region). The
/// min_core_abs floor suppresses spurious cores in large uniform regions; set
/// it to 0 to disable.
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

} // namespace brew::clustering
