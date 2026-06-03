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

        Eigen::MatrixXd F;
        Eigen::VectorXd next_mean;
        Eigen::Matrix3d next_rotation;

        if (!prev.needs_pca_alignment()) {

            Eigen::MatrixXd rot_dyn = prev.rotation();
            F = this->dyn_obj_->get_state_mat(dt, prev.mean(), rot_dyn);
            next_mean = this->dyn_obj_->propagate_state(dt, prev.mean(), rot_dyn);
            next_rotation = this->dyn_obj_->propagate_extent(dt, prev.mean(), rot_dyn);
        } else {
            F = this->dyn_obj_->get_state_mat(dt, prev.mean());
            next_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
            next_rotation = prev.rotation();
        }

        Eigen::MatrixXd F_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
        F_aug.topLeftCorner(n_trans, n_trans) = F;

        Eigen::MatrixXd Q_aug = Eigen::MatrixXd::Zero(n_aug, n_aug);
        Q_aug.topLeftCorner(n_trans, n_trans) = this->process_noise_;
        Q_aug.bottomRightCorner(n_rot, n_rot) = rotation_process_noise_;

        Eigen::MatrixXd next_cov = F_aug * prev.covariance() * F_aug.transpose() + Q_aug;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

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

        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        template_matching::PointCloud meas_cloud(meas_mat);

        const auto psm = icp_runner_.run(
            meas_cloud, predicted.template_id(),
            predicted.rotation(), predicted.needs_pca_alignment());
        last_icp_iterations_ = psm.iterations;

        Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, n_trans);
        for (int i = 0; i < d; ++i) {
            H_pos(i, pos_idx[i]) = 1.0;
        }

        constexpr int m_dim = d + 3;
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

        const int M = static_cast<int>(measurement.size()) / d;
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        Eigen::Vector3d centroid = meas_mat.rowwise().mean();

        Eigen::Vector3d t_pred = predicted.position();

        Eigen::Matrix3d P_pos;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                P_pos(i, j) = predicted.covariance()(pos_idx[i], pos_idx[j]);
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

    void set_template_library(std::shared_ptr<template_matching::TemplateLibrary> lib) {
        icp_runner_.set_template_library(std::move(lib));
    }

    [[nodiscard]] int last_icp_iterations() const { return last_icp_iterations_; }

private:
    Eigen::MatrixXd rotation_process_noise_;
    template_matching::TmIcpRunner icp_runner_;
    mutable int last_icp_iterations_ = 0;
};

}

namespace brew::filters {

template <typename Scalar, int D>
struct default_filter<models::TemplatePose<Scalar, D>> { using type = TmEkf<Scalar, D>; };
}
