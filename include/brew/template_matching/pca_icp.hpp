#pragma once

#include "brew/template_matching/icp_base.hpp"
#include <array>

namespace brew::template_matching {

// @mex icp
// @mex_name PcaIcp
// @mex_namespace template_matching
// @mex_args inner_icp:clone:IcpBase, source_template:mat_pc
// @mex_params max_iterations:int:50, tolerance:double:1e-6, max_correspondence_dist:double:1e10, sigma_sq:double:1.0, trim_fraction:double:1.0
class PcaIcp : public IcpBase {
public:

    PcaIcp(std::unique_ptr<IcpBase> inner_icp, const PointCloud& source_template);

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

    [[nodiscard]] static Eigen::Matrix3d pca_axes(const Eigen::MatrixXd& points);

    void set_pca(const Eigen::Matrix3d& axes, const Eigen::Vector3d& centroid) {
        src_axes_ = axes;
        src_centroid_ = centroid;
    }

private:
    std::unique_ptr<IcpBase> inner_;
    Eigen::Matrix3d src_axes_;
    Eigen::Vector3d src_centroid_;

    [[nodiscard]] static std::array<Eigen::Matrix3d, 8> pca_candidates(
        const Eigen::Matrix3d& src_axes, const Eigen::Matrix3d& tgt_axes);
};

}
