#pragma once

#include "brew/core/assert.hpp"
#include "brew/core/filters/filter.hpp"
#include "brew/core/models/trajectory.hpp"
#include "brew/core/models/template_pose.hpp"
#include "brew/core/template_matching/icp_base.hpp"
#include "brew/core/template_matching/point_cloud.hpp"
#include "brew/core/template_matching/template_library.hpp"
#include "brew/core/template_matching/tm_icp_runner.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace brew::filters {

/// Trajectory EKF for template-matching pose tracking (3D only).
/// Stacks translational kinematic state across time (like TrajectoryGaussianEKF),
/// while rotation lives on SO(3) and is tracked only on the current step.
///
/// Must have a TemplateLibrary set via set_template_library() before correct() is called.
// @mex filter
// @mex_name TrajectoryTmEkf
// @mex_dist TrajectoryTemplatePose
// @mex_setters rotation_process_noise:mat
// @mex_handle_setters icp:IcpBase, template_library:TemplateLibrary

template <int MaxWindow, typename Scalar = double, int D = Eigen::Dynamic>
class TrajectoryTmEkf
    : public Filter<models::Trajectory<models::TemplatePose<Scalar, D>, MaxWindow>> {
public:
    using InnerDist = models::TemplatePose<Scalar, D>;
    using Dist = models::Trajectory<InnerDist, MaxWindow>;
    using TrajectoryType = Dist;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    TrajectoryTmEkf() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TrajectoryTmEkf<MaxWindow, Scalar, D>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->rotation_process_noise_ = rotation_process_noise_;
        c->icp_runner_ = icp_runner_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        Dist result = prev;
        const int sd = prev.state_dim;

        Eigen::VectorXd prev_last_state = prev.get_last_state();
        const auto& prev_tp = prev.current();

        // Dynamics: use the rotation-coupled overloads when the component has
        // been PCA-aligned (prior rotation is meaningful), otherwise fall back
        // to the base 3-arg overloads. Matches pre-library behavior.
        Eigen::VectorXd next_state;
        Eigen::MatrixXd F;
        if (!prev_tp.needs_pca_alignment()) {
            Eigen::MatrixXd rot_dyn = prev_tp.rotation();
            next_state = this->dyn_obj_->propagate_state(dt, prev_last_state, rot_dyn);
            F = this->dyn_obj_->get_state_mat(dt, prev_last_state, rot_dyn);
        } else {
            next_state = this->dyn_obj_->propagate_state(dt, prev_last_state);
            F = this->dyn_obj_->get_state_mat(dt, prev_last_state);
        }

        result.advance_window();

        const int last = result.last_index();
        const int prev_last = last - 1;

        if (prev_last >= 0) {
            for (int k = 0; k < last; ++k) {
                result.cov_at(last, k) = F * result.cov_at(prev_last, k);
                result.cov_at(k, last) = result.cov_at(last, k).transpose();
            }
            result.cov_at(last, last) =
                F * result.cov_at(prev_last, prev_last) * F.transpose()
                + this->process_noise_;
        } else {
            result.cov_at(last, last) = this->process_noise_;
        }

        result.mean_at(last) = next_state;

        // Augmented covariance for the current step's TemplatePose
        const int n_rot = prev_tp.rot_dim();
        const int n_aug = sd + n_rot;
        Eigen::MatrixXd aug_cov = Eigen::MatrixXd::Zero(n_aug, n_aug);
        aug_cov.topLeftCorner(sd, sd) = Eigen::MatrixXd(result.cov_at(last, last));
        const Eigen::MatrixXd& prev_aug = prev_tp.covariance();
        aug_cov.bottomRightCorner(n_rot, n_rot) =
            prev_aug.bottomRightCorner(n_rot, n_rot) + rotation_process_noise_;

        Eigen::Matrix3d next_rotation = this->dyn_obj_->propagate_extent(
            dt, prev_last_state, prev_tp.rotation());
        // with_updated_state preserves template_id / pos_indices / flag via
        // copy — the alignment flag propagates from prev_tp automatically.
        result.history_at(last) = prev_tp.with_updated_state(
            next_state, std::move(aug_cov), std::move(next_rotation));

        return result;
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        BREW_ASSERT(icp_runner_.has_library(),
                    "TrajectoryTmEkf: template library not set");

        constexpr int d = 3;
        const int sd = predicted.state_dim;
        const int last = predicted.last_index();
        const int live = predicted.stacked_size();

        const Eigen::VectorXd prev_state = predicted.get_last_state();
        const auto& pred_tp = predicted.current();
        const int n_rot = pred_tp.rot_dim();
        const auto& pos_idx = pred_tp.pos_indices();

        // Reconstruct measurement as 3×M PointCloud (column-major)
        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        template_matching::PointCloud meas_cloud(meas_mat);

        // Stage A: run ICP. Cold-start vs tracking is encapsulated in the runner.
        const auto psm = icp_runner_.run(
            meas_cloud, pred_tp.template_id(),
            pred_tp.rotation(), pred_tp.needs_pca_alignment());

        // Stage B: fixed-size EKF update using the pseudo-measurement.

        // Position-selection matrix on translational state
        Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, sd);
        for (int i = 0; i < d; ++i) {
            H_pos(i, pos_idx[i]) = 1.0;
        }
        Eigen::Vector3d trans_innov = psm.t - H_pos * prev_state;

        // --- Augmented pose correction on current step ---
        // Unified 6-DoF EKF: cold-start vs tracking is encapsulated in
        // psm.R_ref (see TmEkf for the rationale). No branch needed here.
        const Eigen::MatrixXd& P_aug = pred_tp.covariance();
        const Eigen::MatrixXd& R_meas = this->measurement_noise_;
        const int n_aug = sd + n_rot;

        constexpr int m_dim = d + 3;  // 6
        Eigen::Matrix<double, m_dim, 1> innovation;
        innovation.head<3>() = trans_innov;
        innovation.tail<3>() = psm.rotation_innovation();

        Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
        H_aug.topLeftCorner(d, sd) = H_pos;
        H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::Matrix3d::Identity();

        Eigen::MatrixXd S = H_aug * P_aug * H_aug.transpose() + R_meas;
        S = 0.5 * (S + S.transpose());
        auto S_ldlt = S.ldlt();

        Eigen::MatrixXd K_aug = P_aug * H_aug.transpose()
            * S_ldlt.solve(Eigen::MatrixXd::Identity(m_dim, m_dim));

        const Eigen::VectorXd delta = K_aug * innovation;
        const Eigen::Matrix3d next_rotation = psm.apply_rotation_delta(delta.tail<3>());

        Eigen::MatrixXd I_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
        Eigen::MatrixXd next_aug_cov = (I_aug - K_aug * H_aug) * P_aug;
        next_aug_cov = 0.5 * (next_aug_cov + next_aug_cov.transpose());

        const double mahal = innovation.transpose() * S_ldlt.solve(innovation);
        const double log_det_S = S_ldlt.vectorD().array().abs().log().sum();
        constexpr int innov_dim = m_dim;

        // --- Stacked translational correction on the live slice ---

        Eigen::MatrixXd P_live = predicted.covariance().topLeftCorner(live, live);
        Eigen::VectorXd mean_live = predicted.mean().head(live);

        Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(d, live);
        H_dot.block(0, live - sd, d, sd) = H_pos;

        Eigen::MatrixXd S_trans = H_dot * P_live * H_dot.transpose()
                                + R_meas.topLeftCorner(d, d);
        Eigen::MatrixXd K_stacked = P_live * H_dot.transpose() * S_trans.inverse();
        Eigen::VectorXd new_mean_live = mean_live + K_stacked * trans_innov;
        Eigen::MatrixXd new_P_live = P_live - K_stacked * H_dot * P_live;

        double log_L_pose = -0.5 * (innov_dim * std::log(2.0 * M_PI) + log_det_S + mahal);
        double log_L = psm.log_likelihood + log_L_pose;
        double likelihood = std::exp(std::clamp(log_L, -500.0, 500.0));
        if (!std::isfinite(likelihood) || likelihood <= 0.0) {
            likelihood = 1e-300;
        }

        Dist result = predicted;
        result.mean().head(live) = new_mean_live;
        result.covariance().topLeftCorner(live, live) = new_P_live;

        for (int i = 0; i < predicted.window_size(); ++i) {
            result.history_at(i).mean() = result.mean_at(i);
        }
        // `with_updated_state_aligned` copies the current-step TemplatePose
        // (preserving template_id / pos_indices), overwrites the kinematic
        // fields, and clears the PCA-alignment flag in one step.
        Eigen::VectorXd last_mean = result.mean_at(last);
        result.history_at(last) = result.history_at(last).with_updated_state_aligned(
            std::move(last_mean), std::move(next_aug_cov), next_rotation);

        return { std::move(result), likelihood };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        constexpr int d = 3;
        const auto& tp = predicted.current();
        const auto& pos_idx = tp.pos_indices();

        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        Eigen::Vector3d centroid = meas_mat.rowwise().mean();

        Eigen::Vector3d t_pred = tp.position();

        const auto& P_aug = tp.covariance();
        Eigen::Matrix3d P_pos;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                P_pos(i, j) = P_aug(pos_idx[i], pos_idx[j]);
            }
        }

        Eigen::Vector3d nu = centroid - t_pred;
        return nu.transpose() * P_pos.ldlt().solve(nu);
    }

    void set_rotation_process_noise(Eigen::MatrixXd Q_rot) {
        rotation_process_noise_ = std::move(Q_rot);
    }

    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_runner_.set_icp(std::move(icp));
    }

    /// Set the template library. Must be called before correct().
    void set_template_library(std::shared_ptr<template_matching::TemplateLibrary> lib) {
        icp_runner_.set_template_library(std::move(lib));
    }

    [[nodiscard]] static constexpr int window_size() { return Dist::max_window_size(); }

private:
    Eigen::MatrixXd rotation_process_noise_;
    template_matching::TmIcpRunner icp_runner_;
};

} // namespace brew::filters
