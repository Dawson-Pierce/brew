#pragma once

// Shared ICP machinery used by template-matching filters (TmEkf, TrajectoryTmEkf).
// Owns the base ICP algorithm, the template library, and a cache of PCA-ICP
// wrappers keyed by template id. Produces the fixed-size IcpPseudoMeasurement
// consumed by the filters' correct() step.

#include "brew/template_matching/icp_base.hpp"
#include "brew/template_matching/pca_icp.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include "brew/template_matching/so3.hpp"
#include "brew/template_matching/template_library.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>

namespace brew::template_matching {

/// Fixed-size 6-DoF pose produced by ICP for a template-matching filter.
///
/// `R_ref` is the reference rotation against which the filter's rotation
/// innovation is computed. For tracking components this equals the predicted
/// rotation; for cold-start components it equals `R` itself, which zeroes
/// the rotation innovation and causes the EKF to adopt ICP's rotation as
/// the posterior state — the cold-start-vs-tracking branch lives entirely
/// in this struct, not in the EKF math.
struct IcpPseudoMeasurement {
    Eigen::Vector3d t;
    Eigen::Matrix3d R;
    Eigen::Matrix3d R_ref;
    double log_likelihood;
    int iterations;

    /// Log_SO3(R · R_ref^T). Zero at cold start.
    [[nodiscard]] Eigen::Vector3d rotation_innovation() const {
        return so3::log(Eigen::Matrix3d(R * R_ref.transpose()));
    }

    /// Posterior rotation given the Kalman-update delta in so(3).
    [[nodiscard]] Eigen::Matrix3d apply_rotation_delta(
        const Eigen::Vector3d& delta_rot) const {
        return so3::exp(delta_rot) * R_ref;
    }
};

/// Owns the ICP algorithms + template library + per-id PCA-ICP cache used by
/// both TmEkf and TrajectoryTmEkf. Filters compose this object instead of
/// duplicating its state and methods.
class TmIcpRunner {
public:
    TmIcpRunner() = default;

    void set_icp(std::shared_ptr<IcpBase> icp) {
        icp_ = std::move(icp);
        icp_cache_.clear();
    }

    void set_template_library(std::shared_ptr<TemplateLibrary> lib) {
        template_library_ = std::move(lib);
        icp_cache_.clear();
    }

    [[nodiscard]] bool has_library() const {
        return static_cast<bool>(template_library_);
    }

    /// Run ICP and produce the fixed-size pseudo-measurement.
    ///
    /// Cold start (`cold_start == true`): PCA-ICP seeded from the library's
    /// precomputed PCA axes/centroid. `R_ref` is set to the ICP rotation so
    /// the downstream rotation innovation is zero.
    ///
    /// Tracking (`cold_start == false`): base ICP seeded with `R_pred` and
    /// the cluster centroid. `R_ref = R_pred`.
    ///
    /// Divergence guard: point-to-plane ICP's 6x6 linearized solve can go
    /// rank-deficient on degenerate correspondences, sending the translation
    /// update to absurd values. If ICP lands further than
    /// `max(5 × cluster_diameter, 50m)` from the cluster centroid — or
    /// produces non-finite output — the pseudo-measurement is clamped to
    /// `(cluster_centroid, R_pred)` with a floored log-likelihood so the
    /// filter's state doesn't get corrupted and the PHD weight collapses.
    [[nodiscard]] IcpPseudoMeasurement run(
        const PointCloud& meas,
        int template_id,
        const Eigen::Matrix3d& R_pred,
        bool cold_start) const {

        const Eigen::Vector3d t_init = meas.points().rowwise().mean();
        auto entry = template_library_->get_entry(template_id);

        IcpResult icp_result;
        if (cold_start) {
            const auto& pca_icp = get_or_build_pca_icp(template_id, entry);
            icp_result = pca_icp.align(*entry.cloud, meas, R_pred, t_init);
        } else {
            icp_result = icp_->align(*entry.cloud, meas, R_pred, t_init);
        }

        // Divergence guard
        const Eigen::Vector3d bbox_min = meas.points().rowwise().minCoeff();
        const Eigen::Vector3d bbox_max = meas.points().rowwise().maxCoeff();
        const double cluster_diag = (bbox_max - bbox_min).norm();
        const double drift = (icp_result.translation - t_init).norm();
        const double divergence_threshold = std::max(5.0 * cluster_diag, 50.0);

        if (drift > divergence_threshold
            || !icp_result.translation.allFinite()
            || !icp_result.rotation.allFinite()) {
            return {
                t_init,
                R_pred,
                R_pred,
                -std::numeric_limits<double>::infinity(),
                icp_result.iterations
            };
        }

        const Eigen::Matrix3d R_out = icp_result.rotation;
        const Eigen::Matrix3d R_ref = cold_start ? R_out : R_pred;
        return {
            Eigen::Vector3d(icp_result.translation),
            R_out,
            R_ref,
            icp_result.log_likelihood,
            icp_result.iterations
        };
    }

private:
    /// Lazily construct (and cache) a PcaIcp wrapper for the given template id,
    /// seeded with the library's precomputed PCA axes/centroid. The cache
    /// avoids rebuilding the 8 PCA candidate rotations per call.
    [[nodiscard]] const PcaIcp& get_or_build_pca_icp(
        int id, const TemplateLibrary::Entry& entry) const {
        auto it = icp_cache_.find(id);
        if (it != icp_cache_.end()) return *it->second;

        auto pca = std::make_shared<PcaIcp>(icp_->clone(), *entry.cloud);
        pca->set_pca(entry.pca_axes, entry.pca_centroid);
        icp_cache_[id] = pca;
        return *pca;
    }

    std::shared_ptr<IcpBase> icp_;
    std::shared_ptr<TemplateLibrary> template_library_;
    mutable std::unordered_map<int, std::shared_ptr<PcaIcp>> icp_cache_;
};

} // namespace brew::template_matching
