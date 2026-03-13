#pragma once

#include "brew/template_matching/icp_base.hpp"

namespace brew::template_matching {

/// SVD-based point-to-point ICP (Arun et al. 1987).
class PointToPointIcp : public IcpBase {
public:
    PointToPointIcp() = default;

    [[nodiscard]] IcpResult align(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R_init = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& t_init = Eigen::Vector3d::Zero()) const override;

    [[nodiscard]] double likelihood(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t) const override;

    [[nodiscard]] std::unique_ptr<IcpBase> clone() const override;
};

} // namespace brew::template_matching
