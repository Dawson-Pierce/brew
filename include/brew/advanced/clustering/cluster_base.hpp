#pragma once

#include <Eigen/Dense>
#include <vector>

namespace brew::clustering {

/// Abstract base for clustering objects consumed by RFS filters. A clusterer
/// partitions its input into groups; each returned matrix's columns are the
/// (position + weight) measurements of one cluster, ready for the filter.
class ClusterBase {
public:
    virtual ~ClusterBase() = default;

    [[nodiscard]] virtual std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const = 0;
};

} // namespace brew::clustering
