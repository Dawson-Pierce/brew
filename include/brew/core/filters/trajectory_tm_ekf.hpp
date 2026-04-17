#pragma once

#include "brew/core/filters/filter.hpp"
#include "brew/core/filters/tm_ekf.hpp"  // for detail::Log_SO3, Exp_SO3, Log_SO2, Exp_SO2
#include "brew/core/models/trajectory.hpp"
#include "brew/core/models/template_pose.hpp"
#include "brew/core/template_matching/icp_base.hpp"
#include "brew/core/template_matching/pca_icp.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>

namespace brew::filters {

/// Trajectory EKF for template-matching pose tracking.
/// Stacks translational kinematic state across time (like TrajectoryGaussianEKF),
/// while rotation lives on SO(d) and is tracked only on the current step.
// @mex filter
// @mex_name TrajectoryTmEkf
// @mex_dist TrajectoryTemplatePose
// @mex_setters window_size:int, rotation_process_noise:mat
// @mex_handle_setters icp:IcpBase

template <typename Scalar = double, int D = Eigen::Dynamic, int MaxWindow = Eigen::Dynamic>
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
        auto c = std::make_unique<TrajectoryTmEkf<Scalar, D, MaxWindow>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->rotation_process_noise_ = rotation_process_noise_;
        c->icp_ = icp_;
        c->icp_map_ = icp_map_;
        c->l_window_ = l_window_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        Dist result = prev;
        const int sd = prev.state_dim;

        Eigen::VectorXd prev_last_state = prev.get_last_state();
        const auto& prev_tp = prev.current();

        // Dynamics (with rotation coupling if supported)
        Eigen::VectorXd next_state;
        Eigen::MatrixXd F, G;
        if (prev_tp.has_rotation()) {
            next_state = this->dyn_obj_->propagate_state(dt, prev_last_state, prev_tp.rotation());
            F = this->dyn_obj_->get_state_mat(dt, prev_last_state, prev_tp.rotation());
            // G = this->dyn_obj_->get_input_mat(dt, prev_last_state, prev_tp.rotation());
        } else {
            next_state = this->dyn_obj_->propagate_state(dt, prev_last_state);
            F = this->dyn_obj_->get_state_mat(dt, prev_last_state);
            // G = this->dyn_obj_->get_input_mat(dt, prev_last_state);
        }

        const int cap_hint = Dist::fixed_window ? MaxWindow : l_window_;
        result.advance_window(cap_hint);

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

        // Rotation: propagate from prev's current, or stay empty at cold start
        Eigen::MatrixXd next_rotation;
        if (prev_tp.has_rotation()) {
            next_rotation = this->dyn_obj_->propagate_extent(dt, prev_last_state, prev_tp.rotation());
        }

        // Augmented covariance for the current step's TemplatePose
        const int n_rot = prev_tp.rot_dim();
        const int n_aug = sd + n_rot;
        Eigen::MatrixXd aug_cov = Eigen::MatrixXd::Zero(n_aug, n_aug);
        aug_cov.topLeftCorner(sd, sd) = Eigen::MatrixXd(result.cov_at(last, last));
        const Eigen::MatrixXd& prev_aug = prev_tp.covariance();
        aug_cov.bottomRightCorner(n_rot, n_rot) =
            prev_aug.bottomRightCorner(n_rot, n_rot) + rotation_process_noise_;

        result.history_at(last) = InnerDist(
            next_state, aug_cov, next_rotation,
            prev_tp.template_ptr(), prev_tp.pos_indices());

        return result;
    }

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const Dist& predicted) const override {

        const int sd = predicted.state_dim;
        const int last = predicted.last_index();
        const int live = predicted.stacked_size();

        const Eigen::VectorXd prev_state = predicted.get_last_state();
        const auto& pred_tp = predicted.current();
        const int d = pred_tp.dim();
        const int n_rot = pred_tp.rot_dim();
        const auto& pos_idx = pred_tp.pos_indices();

        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        template_matching::PointCloud meas_cloud(meas_mat);

        const bool cold_start = !pred_tp.has_rotation();
        Eigen::VectorXd t_init = meas_mat.rowwise().mean();

        template_matching::IcpResult icp_result;
        Eigen::VectorXd t_icp(d);
        Eigen::MatrixXd R_icp(d, d);
        const auto& icp_to_use = cold_start
            ? get_icp(&pred_tp.get_template())
            : *icp_;

        if (d == 2) {
            Eigen::Matrix3d R_pred_3d = Eigen::Matrix3d::Identity();
            if (pred_tp.has_rotation())
                R_pred_3d.topLeftCorner(2, 2) = pred_tp.rotation();
            Eigen::Vector3d t_init_3d = Eigen::Vector3d::Zero();
            t_init_3d.head(2) = t_init;

            const auto& templ_pts = pred_tp.get_template().points();
            Eigen::MatrixXd templ_3d = Eigen::MatrixXd::Zero(3, templ_pts.cols());
            templ_3d.topRows(2) = templ_pts;
            template_matching::PointCloud templ_cloud_3d(templ_3d);

            Eigen::MatrixXd meas_3d = Eigen::MatrixXd::Zero(3, M);
            meas_3d.topRows(2) = meas_mat;
            template_matching::PointCloud meas_cloud_3d(meas_3d);

            icp_result = icp_to_use.align(templ_cloud_3d, meas_cloud_3d, R_pred_3d, t_init_3d);

            R_icp = icp_result.rotation.topLeftCorner(2, 2);
            t_icp = icp_result.translation.head(2);
        } else {
            Eigen::Matrix3d R_pred = pred_tp.has_rotation()
                ? Eigen::Matrix3d(pred_tp.rotation())
                : Eigen::Matrix3d::Identity();

            icp_result = icp_to_use.align(
                pred_tp.get_template(), meas_cloud,
                R_pred, Eigen::Vector3d(t_init));
            R_icp = icp_result.rotation;
            t_icp = icp_result.translation;
        }

        // Position-selection matrix on translational state
        Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, sd);
        for (int i = 0; i < d; ++i) {
            H_pos(i, pos_idx[i]) = 1.0;
        }
        Eigen::VectorXd trans_innov = t_icp - H_pos * prev_state;

        // --- Augmented pose correction on current step ---
        Eigen::MatrixXd next_rotation;
        Eigen::MatrixXd next_aug_cov;
        double mahal;
        double log_det_S;
        int innov_dim;

        const Eigen::MatrixXd& P_aug = pred_tp.covariance();
        const Eigen::MatrixXd& R_meas = this->measurement_noise_;

        if (cold_start) {
            Eigen::MatrixXd R_trans = R_meas.topLeftCorner(d, d);
            Eigen::MatrixXd S_t = H_pos * P_aug.topLeftCorner(sd, sd) * H_pos.transpose() + R_trans;
            S_t = 0.5 * (S_t + S_t.transpose());
            auto S_ldlt = S_t.ldlt();

            Eigen::MatrixXd K_t = P_aug.topLeftCorner(sd, sd) * H_pos.transpose()
                * S_ldlt.solve(Eigen::MatrixXd::Identity(d, d));

            next_rotation = R_icp;
            next_aug_cov = P_aug;
            Eigen::MatrixXd P_trans_updated = (Eigen::MatrixXd::Identity(sd, sd) - K_t * H_pos)
                * P_aug.topLeftCorner(sd, sd);
            next_aug_cov.topLeftCorner(sd, sd) = 0.5 * (P_trans_updated + P_trans_updated.transpose());

            mahal = trans_innov.transpose() * S_ldlt.solve(trans_innov);
            log_det_S = S_ldlt.vectorD().array().abs().log().sum();
            innov_dim = d;
        } else {
            Eigen::VectorXd rot_innov(n_rot);
            if (d == 3) {
                Eigen::Matrix3d R_err = R_icp * pred_tp.rotation().transpose();
                rot_innov = detail::Log_SO3(Eigen::Matrix3d(R_err));
            } else {
                Eigen::Matrix2d R_err = R_icp * Eigen::Matrix2d(pred_tp.rotation().transpose());
                rot_innov(0) = detail::Log_SO2(Eigen::Matrix2d(R_err));
            }

            const int m_dim = d + n_rot;
            Eigen::VectorXd innovation(m_dim);
            innovation.head(d) = trans_innov;
            innovation.tail(n_rot) = rot_innov;

            const int n_aug = sd + n_rot;
            Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
            H_aug.topLeftCorner(d, sd) = H_pos;
            H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::MatrixXd::Identity(n_rot, n_rot);

            Eigen::MatrixXd S = H_aug * P_aug * H_aug.transpose() + R_meas;
            S = 0.5 * (S + S.transpose());

            auto S_ldlt = S.ldlt();
            Eigen::MatrixXd K_aug = P_aug * H_aug.transpose() * S.inverse();

            Eigen::VectorXd delta = K_aug * innovation;
            Eigen::VectorXd delta_rot = delta.tail(n_rot);

            if (d == 3) {
                next_rotation = detail::Exp_SO3(Eigen::Vector3d(delta_rot)) * pred_tp.rotation();
            } else {
                next_rotation = detail::Exp_SO2(delta_rot(0)) * Eigen::Matrix2d(pred_tp.rotation());
            }

            Eigen::MatrixXd I_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
            next_aug_cov = (I_aug - K_aug * H_aug) * P_aug;
            next_aug_cov = 0.5 * (next_aug_cov + next_aug_cov.transpose());

            mahal = innovation.transpose() * S_ldlt.solve(innovation);
            log_det_S = S_ldlt.vectorD().array().abs().log().sum();
            innov_dim = m_dim;
        }

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
        double log_L = icp_result.log_likelihood + log_L_pose;
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
        auto& last_tp = result.history_at(last);
        last_tp.mean() = Eigen::VectorXd(result.mean_at(last));
        last_tp.covariance() = next_aug_cov;
        last_tp.rotation() = next_rotation;

        return { std::move(result), likelihood };
    }

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const Dist& predicted) const override {

        const auto& tp = predicted.current();
        const int d = tp.dim();
        const auto& pos_idx = tp.pos_indices();

        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        Eigen::VectorXd centroid = meas_mat.rowwise().mean();

        Eigen::VectorXd t_pred = tp.position();

        const auto& P_aug = tp.covariance();
        Eigen::MatrixXd P_pos(d, d);
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                P_pos(i, j) = P_aug(pos_idx[i], pos_idx[j]);
            }
        }

        Eigen::VectorXd nu = centroid - t_pred;
        return nu.transpose() * P_pos.ldlt().solve(nu);
    }

    void set_rotation_process_noise(Eigen::MatrixXd Q_rot) {
        rotation_process_noise_ = std::move(Q_rot);
    }

    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_ = std::move(icp);
        icp_map_.clear();
    }

    void set_pca_prior(const template_matching::PointCloud* templ,
                       const Eigen::Matrix3d& axes, const Eigen::Vector3d& centroid) {
        pca_prior_map_[templ] = {axes, centroid};
        icp_map_.erase(templ);
    }

    void set_window_size(int L) { l_window_ = L; }

private:
    [[nodiscard]] const template_matching::IcpBase& get_icp(
        const template_matching::PointCloud* templ) const {
        auto it = icp_map_.find(templ);
        if (it != icp_map_.end()) return *it->second;
        if (templ->dim() == 3) {
            auto pca = std::make_shared<template_matching::PcaIcp>(
                icp_->clone(), *templ);
            auto pit = pca_prior_map_.find(templ);
            if (pit != pca_prior_map_.end()) {
                pca->set_pca(pit->second.first, pit->second.second);
            }
            icp_map_[templ] = pca;
            return *pca;
        }
        return *icp_;
    }

    Eigen::MatrixXd rotation_process_noise_;
    std::shared_ptr<template_matching::IcpBase> icp_;
    mutable std::unordered_map<const template_matching::PointCloud*,
                               std::shared_ptr<template_matching::IcpBase>> icp_map_;
    std::unordered_map<const template_matching::PointCloud*,
                       std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pca_prior_map_;
    int l_window_ = 50;
};

} // namespace brew::filters
