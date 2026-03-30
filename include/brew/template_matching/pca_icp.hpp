#pragma once

#include "brew/template_matching/icp_base.hpp"
#include <array>

namespace brew::template_matching {

/// ICP wrapper with PCA pre-alignment for rotation robustness.
/// Precomputes template (source) PCA axes. At alignment time, computes target
/// PCA, generates 8 sign-flip candidates, picks the one closest to R_init
/// (the filter's predicted rotation), and runs ICP once from that.
/// Overhead vs plain ICP: one 3x3 eigendecomposition on the measurement cloud.
class PcaIcp : public IcpBase {
public:
    /// Construct with inner ICP and the source template for PCA precomputation.
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

    /// Compute PCA axes (columns = eigenvectors sorted by descending eigenvalue).
    [[nodiscard]] static Eigen::Matrix3d pca_axes(const Eigen::MatrixXd& points);

    /// Override the precomputed PCA axes and centroid.
    void set_pca(const Eigen::Matrix3d& axes, const Eigen::Vector3d& centroid) {
        src_axes_ = axes;
        src_centroid_ = centroid;
    }

private:
    std::unique_ptr<IcpBase> inner_;
    Eigen::Matrix3d src_axes_;       // precomputed template PCA axes
    Eigen::Vector3d src_centroid_;   // precomputed template centroid

    /// Generate all 8 sign-flip rotation candidates from PCA alignment.
    [[nodiscard]] static std::array<Eigen::Matrix3d, 8> pca_candidates(
        const Eigen::Matrix3d& src_axes, const Eigen::Matrix3d& tgt_axes);
};

} // namespace brew::template_matching
