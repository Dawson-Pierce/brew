#pragma once

#include "brew/core/models/mixture.hpp"
#include "brew/core/models/gaussian.hpp"
#include "brew/core/models/ggiw.hpp"
#include "brew/core/models/ggiw_orientation.hpp"
#include "brew/core/models/template_pose.hpp"
#include "brew/core/models/trajectory.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace brew::fusion {

// ---- Gaussian merge: iterative closest-pair, moment-match ----

/// Merge close Gaussian components via moment matching.
template <typename Scalar, int D, int N>
void merge(models::Mixture<models::Gaussian<Scalar, D>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double min_dist = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                const Eigen::VectorXd diff = mix.component(i).mean() - mix.component(j).mean();
                const Eigen::MatrixXd C = 0.5 * (mix.component(i).covariance() + mix.component(j).covariance());
                const double d2 = diff.transpose() * C.ldlt().solve(diff);
                if (d2 < min_dist) {
                    min_dist = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (min_dist < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double w = wi + wj;

            const Eigen::VectorXd& mi = mix.component(best_i).mean();
            const Eigen::VectorXd& mj = mix.component(best_j).mean();
            const Eigen::MatrixXd& Pi = mix.component(best_i).covariance();
            const Eigen::MatrixXd& Pj = mix.component(best_j).covariance();

            Eigen::VectorXd m_new = (wi * mi + wj * mj) / w;
            Eigen::MatrixXd P_new = (wi * (Pi + (mi - m_new) * (mi - m_new).transpose())
                                    + wj * (Pj + (mj - m_new) * (mj - m_new).transpose())) / w;

            mix.component(best_i).mean() = m_new;
            mix.component(best_i).covariance() = P_new;
            mix.weights()(static_cast<Eigen::Index>(best_i)) = w;
            mix.remove_component(best_j);

            keep_merging = true;
        }
    }
}

// ---- GGIW merge: heaviest-first grouping, Cholesky-gated, weighted avg ----

/// Merge close GGIW components via weighted averaging (heaviest-first grouping).
template <typename Scalar, int D, int De, int N>
void merge(models::Mixture<models::GGIW<Scalar, D, De>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::GGIW<Scalar, D, De>, N> result;

    while (true) {
        // Find heaviest remaining
        double max_w = -1.0;
        std::size_t ref = 0;
        bool found = false;
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

        // Collect group indices within gate
        std::vector<std::size_t> grp;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (!remaining[i]) continue;
            const Eigen::VectorXd d = mix.component(i).mean() - mix.component(ref).mean();
            const Eigen::VectorXd y = llt.matrixL().solve(d);
            if (y.squaredNorm() <= threshold) {
                grp.push_back(i);
            }
        }

        double w_sum = 0.0;
        for (auto idx : grp) w_sum += mix.weight(idx);

        if (w_sum <= 0.0) {
            for (auto idx : grp) remaining[idx] = false;
            continue;
        }

        // Weighted averages of all parameters
        const int n = static_cast<int>(mix.component(ref).mean().size());
        const int d = mix.component(ref).extent_dim();

        Eigen::VectorXd m_new = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(n, n);
        double a_new = 0.0, b_new = 0.0, v_new = 0.0;
        Eigen::MatrixXd V_new = Eigen::MatrixXd::Zero(d, d);

        for (auto idx : grp) {
            const double w = mix.weight(idx);
            m_new += w * mix.component(idx).mean();
            P_new += w * mix.component(idx).covariance();
            a_new += w * mix.component(idx).alpha();
            b_new += w * mix.component(idx).beta();
            v_new += w * mix.component(idx).v();
            V_new += w * mix.component(idx).V();
        }
        m_new /= w_sum;
        P_new /= w_sum;
        P_new = 0.5 * (P_new + P_new.transpose());
        a_new /= w_sum;
        b_new /= w_sum;
        v_new /= w_sum;
        V_new /= w_sum;
        V_new = 0.5 * (V_new + V_new.transpose());

        result.add_component(
            std::make_unique<models::GGIW<Scalar, D, De>>(a_new, b_new, m_new, P_new, v_new, V_new),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

// ---- GGIWOrientation merge: same as GGIW, basis left empty for filter to recompute ----

/// Merge close GGIWOrientation components (same as GGIW merge; basis left empty for filter to recompute).
template <typename Scalar, int D, int De, int N>
void merge(models::Mixture<models::GGIWOrientation<Scalar, D, De>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::GGIWOrientation<Scalar, D, De>, N> result;

    while (true) {
        // Find heaviest remaining
        double max_w = -1.0;
        std::size_t ref = 0;
        bool found = false;
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

        // Collect group indices within gate
        std::vector<std::size_t> grp;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (!remaining[i]) continue;
            const Eigen::VectorXd d = mix.component(i).mean() - mix.component(ref).mean();
            const Eigen::VectorXd y = llt.matrixL().solve(d);
            if (y.squaredNorm() <= threshold) {
                grp.push_back(i);
            }
        }

        double w_sum = 0.0;
        for (auto idx : grp) w_sum += mix.weight(idx);

        if (w_sum <= 0.0) {
            for (auto idx : grp) remaining[idx] = false;
            continue;
        }

        // Weighted averages of all parameters
        const int n = static_cast<int>(mix.component(ref).mean().size());
        const int d = mix.component(ref).extent_dim();

        Eigen::VectorXd m_new = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(n, n);
        double a_new = 0.0, b_new = 0.0, v_new = 0.0;
        Eigen::MatrixXd V_new = Eigen::MatrixXd::Zero(d, d);

        for (auto idx : grp) {
            const double w = mix.weight(idx);
            m_new += w * mix.component(idx).mean();
            P_new += w * mix.component(idx).covariance();
            a_new += w * mix.component(idx).alpha();
            b_new += w * mix.component(idx).beta();
            v_new += w * mix.component(idx).v();
            V_new += w * mix.component(idx).V();
        }
        m_new /= w_sum;
        P_new /= w_sum;
        P_new = 0.5 * (P_new + P_new.transpose());
        a_new /= w_sum;
        b_new /= w_sum;
        v_new /= w_sum;
        V_new /= w_sum;
        V_new = 0.5 * (V_new + V_new.transpose());

        // Basis/eigenvalues left at defaults — filter recomputes on next correction
        result.add_component(
            std::make_unique<models::GGIWOrientation<Scalar, D, De>>(a_new, b_new, m_new, P_new, v_new, V_new),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

// ---- Helper: Mahalanobis distance for full-state trajectories (same-size only) ----

inline double trajectory_mahal_dist(const Eigen::VectorXd& mi, const Eigen::MatrixXd& Pi,
                                    const Eigen::VectorXd& mj, const Eigen::MatrixXd& Pj) {
    if (mi.size() != mj.size()) return std::numeric_limits<double>::infinity();

    Eigen::MatrixXd C = 0.5 * (Pi + Pj);
    C = 0.5 * (C + C.transpose());

    Eigen::LLT<Eigen::MatrixXd> llt(C);
    if (llt.info() != Eigen::Success) {
        const double eps_reg = 1e-9 * std::max(1.0, C.trace() / C.rows());
        C += eps_reg * Eigen::MatrixXd::Identity(C.rows(), C.cols());
        llt.compute(C);
        if (llt.info() != Eigen::Success) {
            // Fallback to pseudoinverse
            const Eigen::VectorXd diff = mi - mj;
            return diff.transpose() * C.completeOrthogonalDecomposition().pseudoInverse() * diff;
        }
    }
    const Eigen::VectorXd y = llt.matrixL().solve(mi - mj);
    return y.squaredNorm();
}

// ---- Trajectory<Gaussian<>> merge: full-state Mahalanobis, keep longer/heavier ----

/// Merge close Trajectory<Gaussian<>> components (same-size, absorb into longer/heavier).
template <typename Scalar, int D, int N, int MaxWindow>
void merge(models::Mixture<models::Trajectory<models::Gaussian<Scalar, D>, MaxWindow>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                const double d2 = trajectory_mahal_dist(
                    mix.component(i).mean(), mix.component(i).covariance(),
                    mix.component(j).mean(), mix.component(j).covariance());
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_d2 < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double W = wi + wj;

            const int len_i = static_cast<int>(mix.component(best_i).mean_history().cols());
            const int len_j = static_cast<int>(mix.component(best_j).mean_history().cols());

            std::size_t keep, drop;
            if (len_i > len_j || (len_i == len_j && wi >= wj)) {
                keep = best_i; drop = best_j;
            } else {
                keep = best_j; drop = best_i;
            }

            // Absorb last-state into keeper
            auto& base = mix.component(keep);
            const int sd = base.state_dim;
            const int n_total = static_cast<int>(base.mean().size());
            const int start = n_total - sd;

            Eigen::VectorXd m = base.mean();
            Eigen::MatrixXd C = base.covariance();
            m.segment(start, sd) = base.get_last_state();
            C.block(start, start, sd, sd) = base.get_last_cov();
            C = 0.5 * (C + C.transpose());
            base.mean() = m;
            base.covariance() = C;

            mix.weights()(static_cast<Eigen::Index>(keep)) = W;
            mix.remove_component(drop);
            keep_merging = true;
        }
    }
}

// ---- Trajectory<GGIW<>> merge: same trajectory merge logic ----

/// Merge close Trajectory<GGIW<>> components (same-size, absorb into longer/heavier).
template <typename Scalar, int D, int De, int N, int MaxWindow>
void merge(models::Mixture<models::Trajectory<models::GGIW<Scalar, D, De>, MaxWindow>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                const double d2 = trajectory_mahal_dist(
                    mix.component(i).mean(), mix.component(i).covariance(),
                    mix.component(j).mean(), mix.component(j).covariance());
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_d2 < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double W = wi + wj;

            const int len_i = static_cast<int>(mix.component(best_i).mean_history().cols());
            const int len_j = static_cast<int>(mix.component(best_j).mean_history().cols());

            std::size_t keep, drop;
            if (len_i > len_j || (len_i == len_j && wi >= wj)) {
                keep = best_i; drop = best_j;
            } else {
                keep = best_j; drop = best_i;
            }

            auto& base = mix.component(keep);
            const int sd = base.state_dim;
            const int n_total = static_cast<int>(base.mean().size());
            const int start = n_total - sd;

            Eigen::VectorXd m = base.mean();
            Eigen::MatrixXd C = base.covariance();
            m.segment(start, sd) = base.get_last_state();
            C.block(start, start, sd, sd) = base.get_last_cov();
            C = 0.5 * (C + C.transpose());
            base.mean() = m;
            base.covariance() = C;

            mix.weights()(static_cast<Eigen::Index>(keep)) = W;
            mix.remove_component(drop);
            keep_merging = true;
        }
    }
}

// ---- Trajectory<GGIWOrientation<>> merge: same trajectory merge logic, basis left empty ----

/// Merge close Trajectory<GGIWOrientation<>> components (same-size, absorb into longer/heavier; basis left empty).
template <typename Scalar, int D, int De, int N, int MaxWindow>
void merge(models::Mixture<models::Trajectory<models::GGIWOrientation<Scalar, D, De>, MaxWindow>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                const double d2 = trajectory_mahal_dist(
                    mix.component(i).mean(), mix.component(i).covariance(),
                    mix.component(j).mean(), mix.component(j).covariance());
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_d2 < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double W = wi + wj;

            const int len_i = static_cast<int>(mix.component(best_i).mean_history().cols());
            const int len_j = static_cast<int>(mix.component(best_j).mean_history().cols());

            std::size_t keep, drop;
            if (len_i > len_j || (len_i == len_j && wi >= wj)) {
                keep = best_i; drop = best_j;
            } else {
                keep = best_j; drop = best_i;
            }

            auto& base = mix.component(keep);
            const int sd = base.state_dim;
            const int n_total = static_cast<int>(base.mean().size());
            const int start = n_total - sd;

            Eigen::VectorXd m = base.mean();
            Eigen::MatrixXd C = base.covariance();
            m.segment(start, sd) = base.get_last_state();
            C.block(start, start, sd, sd) = base.get_last_cov();
            C = 0.5 * (C + C.transpose());
            base.mean() = m;
            base.covariance() = C;

            mix.weights()(static_cast<Eigen::Index>(keep)) = W;
            mix.remove_component(drop);
            keep_merging = true;
        }
    }
}

// ---- TemplatePose merge: heaviest-first, same-template only, rotation via SVD ----

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

// ---- Trajectory<TemplatePose<>> merge: same-size, same-template, absorb into longer/heavier ----

/// Merge close Trajectory<TemplatePose<>> components (same-size, same-template, absorb into longer/heavier).
template <typename Scalar, int D, int N, int MaxWindow>
void merge(models::Mixture<models::Trajectory<models::TemplatePose<Scalar, D>, MaxWindow>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                // Same-template constraint
                if (mix.component(i).current().template_id() !=
                    mix.component(j).current().template_id()) continue;

                // Rotation distance gate: skip if rotations are > 90° apart,
                // but only when both sides have been PCA-aligned. A birth
                // component's identity rotation should not block absorption.
                if (!mix.component(i).current().needs_pca_alignment()
                    && !mix.component(j).current().needs_pca_alignment()) {
                    const auto& Ri = mix.component(i).current().rotation();
                    const auto& Rj = mix.component(j).current().rotation();
                    Eigen::MatrixXd R_err = Ri.transpose() * Rj;
                    double cos_a = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
                    if (std::acos(cos_a) > M_PI / 2.0) continue;
                }

                const double d2 = trajectory_mahal_dist(
                    mix.component(i).mean(), mix.component(i).covariance(),
                    mix.component(j).mean(), mix.component(j).covariance());
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_d2 < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double W = wi + wj;

            const int len_i = static_cast<int>(mix.component(best_i).mean_history().cols());
            const int len_j = static_cast<int>(mix.component(best_j).mean_history().cols());

            std::size_t keep, drop;
            if (len_i > len_j || (len_i == len_j && wi >= wj)) {
                keep = best_i; drop = best_j;
            } else {
                keep = best_j; drop = best_i;
            }

            auto& base = mix.component(keep);
            const int sd = base.state_dim;
            const int n_total = static_cast<int>(base.mean().size());
            const int start = n_total - sd;

            Eigen::VectorXd m = base.mean();
            Eigen::MatrixXd C = base.covariance();
            m.segment(start, sd) = base.get_last_state();
            C.block(start, start, sd, sd) = base.get_last_cov();
            C = 0.5 * (C + C.transpose());
            base.mean() = m;
            base.covariance() = C;

            mix.weights()(static_cast<Eigen::Index>(keep)) = W;
            mix.remove_component(drop);
            keep_merging = true;
        }
    }
}

} // namespace brew::fusion
