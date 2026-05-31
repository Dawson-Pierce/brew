#pragma once

#include "brew/template_matching/icp_base.hpp"

namespace brew::template_matching {

/// SVD-based point-to-point ICP (Arun et al. 1987).
// @mex icp
// @mex_name PointToPointIcp
// @mex_namespace template_matching
// @mex_params max_iterations:int:50, tolerance:double:1e-6, max_correspondence_dist:double:1e10, sigma_sq:double:1.0, trim_fraction:double:1.0
class PointToPointIcp : public IcpBase {
public:
    PointToPointIcp() = default;

    [[nodiscard]] IcpResult align(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R_init = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& t_init = Eigen::Vector3d::Zero()) const override;

    [[nodiscard]] double log_likelihood(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t) const override;

    [[nodiscard]] std::unique_ptr<IcpBase> clone() const override;
};

} // namespace brew::template_matching
