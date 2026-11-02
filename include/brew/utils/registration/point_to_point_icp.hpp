#pragma once

#include "brew/utils/registration/icp_base.hpp"

namespace brew::utils::registration {

// @mex icp
// @mex_name PointToPointIcp
// @mex_namespace utils::registration
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

}
