#pragma once

// Component merge for the `template_pose` package -- owns ONLY the TemplatePose merge
// (no cross-model overload set). In namespace brew::template_pose so the package's
// concrete RFS resolve an unqualified merge(...) here.
#include "brew/shared/mixture.hpp"
#include "brew/template_pose/template_pose_model.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::template_pose {

/// Merge close TemplatePose components (same-template only, rotation averaged via SVD projection).
template <typename Scalar, int D, int N>
void merge(models::Mixture<models::TemplatePose<Scalar, D>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::TemplatePose<Scalar, D>, N> result;

    while (true) {
        // Find heaviest remaining
        double max_w = -1.0;
        std::size_t ref = 0;
        bool found = false;
        // Find max weight of remaining mixture
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (remaining[i] && mix.weight(i) > max_w) {
                max_w = mix.weight(i);
                ref = i;
                found = true;
            }
        }
        if (!found) break;

        // Cholesky of reference covariance for gating
        Eigen::MatrixXd C_ref = mix.component(ref).covariance();
        C_ref = 0.5 * (C_ref + C_ref.transpose());
        Eigen::LLT<Eigen::MatrixXd> llt(C_ref);
        if (llt.info() != Eigen::Success) {
            const double eps_reg = 1e-9 * std::max(1.0, C_ref.trace() / C_ref.rows());
            C_ref += eps_reg * Eigen::MatrixXd::Identity(C_ref.rows(), C_ref.cols());
            llt.compute(C_ref);
        }

        // Collect group: same template + within gate
        std::vector<std::size_t> grp;
        const int ref_templ_id = mix.component(ref).template_id();
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (!remaining[i]) continue;
            // Same-template constraint
            if (mix.component(i).template_id() != ref_templ_id) continue;

            // Augmented state distance (mean + rotation error states share covariance)
            const Eigen::VectorXd d = mix.component(i).mean() - mix.component(ref).mean();

            // Gate on translational state only (rotation is on manifold)
            // Pad to augmented size for Cholesky solve
            const int n_aug = mix.component(ref).aug_dim();
            const int n_trans = mix.component(ref).trans_dim();
            Eigen::VectorXd d_aug = Eigen::VectorXd::Zero(n_aug);
            d_aug.head(n_trans) = d;
            const Eigen::VectorXd y = llt.matrixL().solve(d_aug);
            if (y.squaredNorm() > threshold) continue;

            // Also gate on rotation: don't merge components with very different
            // rotations — but only when both sides have been PCA-aligned
            // (i.e. their rotations are meaningful). A birth component's
            // rotation is a placeholder identity that should not prevent it
            // from being absorbed into a tracked component.
            if (!mix.component(i).needs_pca_alignment()
                && !mix.component(ref).needs_pca_alignment()) {
                const auto& Ri = mix.component(i).rotation();
                const auto& Rr = mix.component(ref).rotation();
                Eigen::MatrixXd R_err = Ri.transpose() * Rr;
                double cos_angle = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
                double rot_dist = std::acos(cos_angle);  // angle in [0, π]
                if (rot_dist > M_PI / 2.0) continue;  // reject if > 90° apart
            }

            grp.push_back(i);
        }

        double w_sum = 0.0;
        for (auto idx : grp) w_sum += mix.weight(idx);

        if (w_sum <= 0.0) {
            for (auto idx : grp) remaining[idx] = false;
            continue;
        }

        const int n_trans = mix.component(ref).trans_dim();
        const int n_aug = mix.component(ref).aug_dim();
        const int d = mix.component(ref).dim();

        // Merged translational mean
        Eigen::VectorXd m_new = Eigen::VectorXd::Zero(n_trans);
        for (auto idx : grp) {
            m_new += mix.weight(idx) * mix.component(idx).mean();
        }
        m_new /= w_sum;

        // Merged rotation: projected arithmetic mean via SVD.
        // Rotation is always present (identity at birth), so every group member
        // contributes.
        Eigen::MatrixXd R_sum = Eigen::MatrixXd::Zero(d, d);
        for (auto idx : grp) {
            const double w = mix.weight(idx);
            R_sum += w * mix.component(idx).rotation();
        }
        R_sum /= w_sum;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(R_sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        // Ensure proper rotation (det = +1)
        Eigen::MatrixXd D_mat = Eigen::MatrixXd::Identity(d, d);
        D_mat(d - 1, d - 1) = (U * V.transpose()).determinant();
        Eigen::MatrixXd R_new = U * D_mat * V.transpose();

        // Moment-matched covariance with translational spread-of-means
        Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(n_aug, n_aug);
        for (auto idx : grp) {
            const double w = mix.weight(idx);
            const Eigen::VectorXd dm = mix.component(idx).mean() - m_new;

            // Spread term only in the translational block
            Eigen::MatrixXd spread = Eigen::MatrixXd::Zero(n_aug, n_aug);
            spread.topLeftCorner(n_trans, n_trans) = dm * dm.transpose();

            P_new += w * (mix.component(idx).covariance() + spread);
        }
        P_new /= w_sum;
        P_new = 0.5 * (P_new + P_new.transpose());

        // Pick the "most aligned" contributor as the base so the merged
        // component inherits its PCA-alignment flag (and any other
        // carry-through state). An already-aligned contributor always wins
        // over a still-cold-start birth; otherwise ref is fine.
        std::size_t base_idx = ref;
        for (auto idx : grp) {
            if (!mix.component(idx).needs_pca_alignment()) { base_idx = idx; break; }
        }
        auto merged = std::make_unique<models::TemplatePose<Scalar, D>>(
            mix.component(base_idx).with_updated_state(m_new, P_new, R_new));

        result.add_component(std::move(merged), w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

}  // namespace brew::template_pose
