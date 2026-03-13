#pragma once

#include "brew/template_matching/icp_base.hpp"

namespace brew::template_matching {

/// Linearized point-to-plane ICP.
class PointToPlaneIcp : public IcpBase {
public:
    PointToPlaneIcp() = default;

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

    /// Estimate surface normals via local PCA with k nearest neighbors.
    [[nodiscard]] static Eigen::MatrixXd estimate_normals(
        const PointCloud& cloud, int k);
};

} // namespace brew::template_matching
