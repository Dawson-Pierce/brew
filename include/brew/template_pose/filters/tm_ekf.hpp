#pragma once
#include "brew/shared/filter_traits.hpp"

#include "brew/assert.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/template_pose/template_pose_model.hpp"
#include "brew/template_matching/icp_base.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include "brew/template_matching/template_library.hpp"
#include "brew/template_matching/tm_icp_runner.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace brew::filters {

/// Template Matching Extended Kalman Filter (3D only).
/// Invariant EKF for pose estimation using ICP-based template matching.
///
/// Must have a TemplateLibrary set via set_template_library() before correct() is called.
// @mex filter
// @mex_name TmEkf
// @mex_dist TemplatePose
// @mex_setters rotation_process_noise:mat
// @mex_handle_setters icp:IcpBase, template_library:TemplateLibrary

template <typename Scalar = double, int D = Eigen::Dynamic>
class TmEkf : public Filter<models::TemplatePose<Scalar, D>> {
public:
    using Dist = models::TemplatePose<Scalar, D>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    TmEkf() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TmEkf<Scalar, D>>();
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

        const int n_trans = prev.trans_dim();
        const int n_rot = prev.rot_dim();
        const int n_aug = prev.aug_dim();

        // Propagate translational state and rotation via dynamics.
        // When the component already has a PCA-aligned prior rotation, use
        // the rotation-coupled dynamics overloads (matches pre-library behavior).
        // Otherwise (birth components needing PCA alignment) use the base
        // 3-arg overloads.
        Eigen::MatrixXd F;
        Eigen::VectorXd next_mean;
        Eigen::Matrix3d next_rotation;

        if (!prev.needs_pca_alignment()) {
            // Rotation is fixed 3x3; DynamicsBase RotationMatrix is dynamic —
            // explicit conversion disambiguates the 3-arg vs 4-arg overload.
            Eigen::MatrixXd rot_dyn = prev.rotation();
            F = this->dyn_obj_->get_state_mat(dt, prev.mean(), rot_dyn);
            next_mean = this->dyn_obj_->propagate_state(dt, prev.mean(), rot_dyn);
            next_rotation = this->dyn_obj_->propagate_extent(dt, prev.mean(), rot_dyn);
        } else {
            F = this->dyn_obj_->get_state_mat(dt, prev.mean());
            next_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
            next_rotation = prev.rotation();  // stays at birth identity
        }

        // Build augmented F: [F, 0; 0, I_rot]
        Eigen::MatrixXd F_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
        F_aug.topLeftCorner(n_trans, n_trans) = F;

        // Build augmented process noise
        Eigen::MatrixXd Q_aug = Eigen::MatrixXd::Zero(n_aug, n_aug);
        Q_aug.topLeftCorner(n_trans, n_trans) = this->process_noise_;
        Q_aug.bottomRightCorner(n_rot, n_rot) = rotation_process_noise_;

        // Propagate augmented covariance
        Eigen::MatrixXd next_cov = F_aug * prev.covariance() * F_aug.transpose() + Q_aug;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        // with_updated_state preserves template_id, pos_indices, and the
        // PCA-alignment flag via copy — no explicit flag propagation needed.
        return prev.with_updated_state(
            std::move(next_mean), std::move(next_cov), std::move(next_rotation));
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        BREW_ASSERT(icp_runner_.has_library(), "TmEkf: template library not set");

        constexpr int d = 3;
        const int n_trans = predicted.trans_dim();
        const int n_rot = predicted.rot_dim();
        const int n_aug = predicted.aug_dim();
        const auto& pos_idx = predicted.pos_indices();

        // Reconstruct measurement as 3×M PointCloud (column-major)
        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        template_matching::PointCloud meas_cloud(meas_mat);

        // Stage A: run ICP, producing a fixed-size pseudo-measurement (t, R).
        // The cold-start-vs-tracking branch lives entirely inside TmIcpRunner.
        const auto psm = icp_runner_.run(
            meas_cloud, predicted.template_id(),
            predicted.rotation(), predicted.needs_pca_alignment());
        last_icp_iterations_ = psm.iterations;

        // Stage B: unified 6-DoF EKF update. For cold start R_ref == R, so
        // psm.rotation_innovation() is zero and psm.apply_rotation_delta
        // returns R unchanged — the filter adopts ICP's rotation without any
        // special case in the EKF math.

        // Build H_pos: selects position indices from translational state
        Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, n_trans);
        for (int i = 0; i < d; ++i) {
            H_pos(i, pos_idx[i]) = 1.0;
        }

        constexpr int m_dim = d + 3;  // 6
        Eigen::Matrix<double, m_dim, 1> innovation;
        innovation.head<3>() = psm.t - H_pos * predicted.mean();
        innovation.tail<3>() = psm.rotation_innovation();

        Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
        H_aug.topLeftCorner(d, n_trans) = H_pos;
        H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::Matrix3d::Identity();

        const Eigen::MatrixXd& P = predicted.covariance();
        Eigen::MatrixXd S = H_aug * P * H_aug.transpose() + this->measurement_noise_;
        S = 0.5 * (S + S.transpose());
        auto S_ldlt = S.ldlt();

        Eigen::MatrixXd K = P * H_aug.transpose()
            * S_ldlt.solve(Eigen::MatrixXd::Identity(m_dim, m_dim));

        const Eigen::VectorXd delta = K * innovation;
        Eigen::VectorXd next_mean = predicted.mean() + delta.head(n_trans);
        Eigen::Matrix3d next_rotation = psm.apply_rotation_delta(delta.tail<3>());

        Eigen::MatrixXd I_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
        Eigen::MatrixXd next_cov = (I_aug - K * H_aug) * P;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        const double mahal = innovation.transpose() * S_ldlt.solve(innovation);
        const double log_det_S = S_ldlt.vectorD().array().abs().log().sum();
        const double log_L_pose = -0.5 * (m_dim * std::log(2.0 * M_PI) + log_det_S + mahal);
        const double log_L = psm.log_likelihood + log_L_pose;
        double likelihood = std::exp(std::clamp(log_L, -500.0, 500.0));
        if (!std::isfinite(likelihood) || likelihood <= 0.0) {
            likelihood = 1e-300;
        }

        // `with_updated_state_aligned` copies `predicted` (preserving
        // template_id, pos_indices, and all other non-kinematic state),
        // overwrites the kinematic fields, and clears the PCA-alignment flag
        // in one move — the flag transitions to "aligned" atomically with
        // the state update, which is exactly the semantic of a correct().
        return {
            predicted.with_updated_state_aligned(
                std::move(next_mean), std::move(next_cov), std::move(next_rotation)),
            likelihood
        };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        constexpr int d = 3;
        const auto& pos_idx = predicted.pos_indices();

        // Centroid of measurement cloud
        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        Eigen::Vector3d centroid = meas_mat.rowwise().mean();

        // Predicted position
        Eigen::Vector3d t_pred = predicted.position();

        // Position-position covariance subblock
        Eigen::Matrix3d P_pos;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                P_pos(i, j) = predicted.covariance()(pos_idx[i], pos_idx[j]);
            }
        }

        Eigen::Vector3d nu = centroid - t_pred;
        return nu.transpose() * P_pos.ldlt().solve(nu);
    }

    // ---- TM-EKF-specific configuration ----
    void set_rotation_process_noise(Eigen::MatrixXd Q_rot) {
        rotation_process_noise_ = std::move(Q_rot);
    }

    /// Set the inner ICP algorithm. PCA wrapping is applied automatically
    /// per template id on first use — no manual per-template registration needed.
    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_runner_.set_icp(std::move(icp));
    }

    /// Set the template library. Must be called before correct().
    void set_template_library(std::shared_ptr<template_matching::TemplateLibrary> lib) {
        icp_runner_.set_template_library(std::move(lib));
    }

    /// ICP iterations from the last correct() call.
    [[nodiscard]] int last_icp_iterations() const { return last_icp_iterations_; }

private:
    Eigen::MatrixXd rotation_process_noise_;
    template_matching::TmIcpRunner icp_runner_;
    mutable int last_icp_iterations_ = 0;
};

} // namespace brew::filters

namespace brew::filters {
// Concrete filter used for this model (RFS devirtualization).
template <typename Scalar, int D>
struct default_filter<models::TemplatePose<Scalar, D>> { using type = TmEkf<Scalar, D>; };
}  // namespace brew::filters
