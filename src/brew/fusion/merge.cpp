#include "brew/fusion/merge.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::fusion {

// ---- Gaussian merge: iterative closest-pair, moment-match ----

void merge(models::Mixture<models::Gaussian>& mix, double threshold) {
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

void merge(models::Mixture<models::GGIW>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::GGIW> result;

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
            std::make_unique<models::GGIW>(m_new, P_new, a_new, b_new, v_new, V_new),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

// ---- GGIWOrientation merge: same as GGIW, basis left empty for filter to recompute ----

void merge(models::Mixture<models::GGIWOrientation>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::GGIWOrientation> result;

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
            std::make_unique<models::GGIWOrientation>(m_new, P_new, a_new, b_new, v_new, V_new),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

// ---- Helper: Mahalanobis distance for full-state trajectories (same-size only) ----

static double trajectory_mahal_dist(const Eigen::VectorXd& mi, const Eigen::MatrixXd& Pi,
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

// ---- Trajectory<Gaussian> merge: full-state Mahalanobis, keep longer/heavier ----

void merge(models::Mixture<models::Trajectory<models::Gaussian>>& mix, double threshold) {
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

// ---- Trajectory<GGIW> merge: same trajectory merge logic ----

void merge(models::Mixture<models::Trajectory<models::GGIW>>& mix, double threshold) {
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

// ---- Trajectory<GGIWOrientation> merge: same trajectory merge logic, basis left empty ----

void merge(models::Mixture<models::Trajectory<models::GGIWOrientation>>& mix, double threshold) {
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

void merge(models::Mixture<models::TemplatePose>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::TemplatePose> result;

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

        // Collect group: same template + within gate
        std::vector<std::size_t> grp;
        const auto ref_templ = mix.component(ref).template_ptr();
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (!remaining[i]) continue;
            // Same-template constraint
            if (mix.component(i).template_ptr() != ref_templ) continue;

            // Augmented state distance (mean + rotation error states share covariance)
            const Eigen::VectorXd d = mix.component(i).mean() - mix.component(ref).mean();

            // Gate on translational state only (rotation is on manifold)
            // Pad to augmented size for Cholesky solve
            const int n_aug = mix.component(ref).aug_dim();
            const int n_trans = mix.component(ref).trans_dim();
            Eigen::VectorXd d_aug = Eigen::VectorXd::Zero(n_aug);
            d_aug.head(n_trans) = d;
            const Eigen::VectorXd y = llt.matrixL().solve(d_aug);
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

        const int n_trans = mix.component(ref).trans_dim();
        const int n_aug = mix.component(ref).aug_dim();
        const int d = mix.component(ref).dim();

        // Weighted average of translational state
        Eigen::VectorXd m_new = Eigen::VectorXd::Zero(n_trans);
        Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(n_aug, n_aug);
        Eigen::MatrixXd R_sum = Eigen::MatrixXd::Zero(d, d);

        bool any_has_rotation = false;
        for (auto idx : grp) {
            const double w = mix.weight(idx);
            m_new += w * mix.component(idx).mean();
            P_new += w * mix.component(idx).covariance();
            if (mix.component(idx).has_rotation()) {
                R_sum += w * mix.component(idx).rotation();
                any_has_rotation = true;
            }
        }
        m_new /= w_sum;
        P_new /= w_sum;
        P_new = 0.5 * (P_new + P_new.transpose());

        // Rotation: projected arithmetic mean via SVD (skip if no rotations)
        Eigen::MatrixXd R_new;
        if (any_has_rotation) {
            R_sum /= w_sum;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(R_sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::MatrixXd U = svd.matrixU();
            Eigen::MatrixXd V = svd.matrixV();
            // Ensure proper rotation (det = +1)
            Eigen::MatrixXd D_mat = Eigen::MatrixXd::Identity(d, d);
            D_mat(d - 1, d - 1) = (U * V.transpose()).determinant();
            R_new = U * D_mat * V.transpose();
        }

        result.add_component(
            std::make_unique<models::TemplatePose>(
                m_new, P_new, R_new, ref_templ,
                mix.component(ref).pos_indices()),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

// ---- Trajectory<TemplatePose> merge: same-size, same-template, absorb into longer/heavier ----

void merge(models::Mixture<models::Trajectory<models::TemplatePose>>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                // Same-template constraint
                if (mix.component(i).current().template_ptr() !=
                    mix.component(j).current().template_ptr()) continue;

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
