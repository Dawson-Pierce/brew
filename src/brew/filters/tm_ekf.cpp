#include "brew/filters/tm_ekf.hpp"
#include <cmath>

namespace brew::filters {

// ---- Lie group helpers ----

static Eigen::Vector3d Log_SO3(const Eigen::Matrix3d& R) {
    const double cos_angle = std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);

    if (angle < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    if (angle > M_PI - 1e-6) {
        // Near π: sin(angle) ≈ 0, so the skew-symmetric extraction fails.
        // Instead, find the rotation axis from R + I (eigenvector with eigenvalue +1).
        Eigen::Matrix3d M = R + Eigen::Matrix3d::Identity();
        // Pick the column of M with the largest norm as the axis
        Eigen::Vector3d axis = M.col(0);
        for (int i = 1; i < 3; ++i) {
            if (M.col(i).squaredNorm() > axis.squaredNorm())
                axis = M.col(i);
        }
        axis.normalize();
        return axis * M_PI;
    }

    Eigen::Vector3d axis;
    axis << R(2, 1) - R(1, 2),
            R(0, 2) - R(2, 0),
            R(1, 0) - R(0, 1);
    axis *= angle / (2.0 * std::sin(angle));
    return axis;
}

static Eigen::Matrix3d Exp_SO3(const Eigen::Vector3d& phi) {
    const double angle = phi.norm();
    if (angle < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d axis = phi / angle;
    Eigen::Matrix3d K;
    K <<     0.0, -axis(2),  axis(1),
         axis(2),      0.0, -axis(0),
        -axis(1),  axis(0),      0.0;

    return Eigen::Matrix3d::Identity()
           + std::sin(angle) * K
           + (1.0 - std::cos(angle)) * K * K;
}

static double Log_SO2(const Eigen::Matrix2d& R) {
    return std::atan2(R(1, 0), R(0, 0));
}

static Eigen::Matrix2d Exp_SO2(double theta) {
    Eigen::Matrix2d R;
    R << std::cos(theta), -std::sin(theta),
         std::sin(theta),  std::cos(theta);
    return R;
}

// ---- Filter implementation ----

std::unique_ptr<Filter<models::TemplatePose>> TmEkf::clone() const {
    auto c = std::make_unique<TmEkf>();
    c->dyn_obj_ = dyn_obj_;
    c->h_ = h_;
    c->H_ = H_;
    c->process_noise_ = process_noise_;
    c->measurement_noise_ = measurement_noise_;
    c->rotation_process_noise_ = rotation_process_noise_;
    c->icp_ = icp_;
    c->icp_map_ = icp_map_;
    return c;
}

models::TemplatePose TmEkf::predict(
    double dt,
    const models::TemplatePose& prev) const {

    const int n_trans = prev.trans_dim();
    const int n_rot = prev.rot_dim();
    const int n_aug = prev.aug_dim();

    // Propagate translational state via dynamics (with rotation coupling if supported)
    Eigen::MatrixXd F, G;
    Eigen::VectorXd next_mean;
    Eigen::MatrixXd next_rotation;

    if (prev.has_rotation()) {
        F = dyn_obj_->get_state_mat(dt, prev.mean(), prev.rotation());
        G = dyn_obj_->get_input_mat(dt, prev.mean(), prev.rotation());
        next_mean = dyn_obj_->propagate_state(dt, prev.mean(), prev.rotation());
        next_rotation = dyn_obj_->propagate_extent(dt, prev.mean(), prev.rotation());
    } else {
        // No rotation yet (birth component) — use base dynamics
        F = dyn_obj_->get_state_mat(dt, prev.mean());
        G = dyn_obj_->get_input_mat(dt, prev.mean());
        next_mean = dyn_obj_->propagate_state(dt, prev.mean());
        next_rotation = Eigen::MatrixXd();  // stays empty
    }

    // Build augmented F: [F, 0; 0, I_rot]
    Eigen::MatrixXd F_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
    F_aug.topLeftCorner(n_trans, n_trans) = F;

    // Build augmented process noise
    Eigen::MatrixXd Q_aug = Eigen::MatrixXd::Zero(n_aug, n_aug);
    Q_aug.topLeftCorner(n_trans, n_trans) = G * process_noise_ * G.transpose();
    Q_aug.bottomRightCorner(n_rot, n_rot) = rotation_process_noise_;

    // Propagate augmented covariance
    Eigen::MatrixXd next_cov = F_aug * prev.covariance() * F_aug.transpose() + Q_aug;
    next_cov = 0.5 * (next_cov + next_cov.transpose());

    return models::TemplatePose(
        std::move(next_mean), std::move(next_cov), std::move(next_rotation),
        prev.template_ptr(), prev.pos_indices());
}

TmEkf::CorrectionResult TmEkf::correct(
    const Eigen::VectorXd& measurement,
    const models::TemplatePose& predicted) const {

    const int d = predicted.dim();
    const int n_trans = predicted.trans_dim();
    const int n_rot = predicted.rot_dim();
    const int n_aug = predicted.aug_dim();
    const auto& pos_idx = predicted.pos_indices();

    // Reconstruct measurement as PointCloud (d×M, column-major)
    const int M = static_cast<int>(measurement.size()) / d;
    Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
    template_matching::PointCloud meas_cloud(meas_mat);

    // Use measurement centroid as ICP initial translation.
    const bool cold_start = !predicted.has_rotation();
    Eigen::VectorXd t_init = meas_mat.rowwise().mean();

    // Run ICP. Use PCA-ICP only for cold start (no rotation yet);
    // during tracking, use the base ICP seeded with the filter's predicted rotation.
    last_icp_iterations_ = 0;
    template_matching::IcpResult icp_result;
    Eigen::VectorXd t_icp(d);
    Eigen::MatrixXd R_icp(d, d);

    // Pick ICP variant: PCA for cold start, base ICP for tracking.
    const auto& icp_to_use = cold_start
        ? get_icp(&predicted.get_template())
        : *icp_;

    if (d == 2) {
        // Embed 2D → 3D
        Eigen::Matrix3d R_pred_3d = Eigen::Matrix3d::Identity();
        if (predicted.has_rotation())
            R_pred_3d.topLeftCorner(2, 2) = predicted.rotation();
        Eigen::Vector3d t_init_3d = Eigen::Vector3d::Zero();
        t_init_3d.head(2) = t_init;

        // Embed template
        const auto& templ_pts = predicted.get_template().points();
        Eigen::MatrixXd templ_3d = Eigen::MatrixXd::Zero(3, templ_pts.cols());
        templ_3d.topRows(2) = templ_pts;
        template_matching::PointCloud templ_cloud_3d(templ_3d);

        // Embed measurement
        Eigen::MatrixXd meas_3d = Eigen::MatrixXd::Zero(3, M);
        meas_3d.topRows(2) = meas_mat;
        template_matching::PointCloud meas_cloud_3d(meas_3d);

        icp_result = icp_to_use.align(templ_cloud_3d, meas_cloud_3d, R_pred_3d, t_init_3d);

        // Project back to 2D
        R_icp = icp_result.rotation.topLeftCorner(2, 2);
        t_icp = icp_result.translation.head(2);
    } else {
        Eigen::Matrix3d R_pred = predicted.has_rotation()
            ? Eigen::Matrix3d(predicted.rotation())
            : Eigen::Matrix3d::Identity();

        icp_result = icp_to_use.align(
            predicted.get_template(), meas_cloud,
            R_pred, Eigen::Vector3d(t_init));

        R_icp = icp_result.rotation;
        t_icp = icp_result.translation;
    }

    last_icp_iterations_ = icp_result.iterations;

    // Build H_pos: selects position indices from translational state
    Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, n_trans);
    for (int i = 0; i < d; ++i) {
        H_pos(i, pos_idx[i]) = 1.0;
    }

    // Translation innovation
    Eigen::VectorXd trans_innov = t_icp - H_pos * predicted.mean();

    Eigen::VectorXd next_mean;
    Eigen::MatrixXd next_rotation;
    Eigen::MatrixXd next_cov;
    double mahal;
    double log_det_S;
    int innov_dim;

    if (!predicted.has_rotation()) {
        // Cold start: no prior rotation. Use ICP rotation directly, standard
        // EKF on translation only, skip rotation innovation.
        const Eigen::MatrixXd& P = predicted.covariance();
        Eigen::MatrixXd R_trans = measurement_noise_.topLeftCorner(d, d);
        Eigen::MatrixXd S_trans = H_pos * P.topLeftCorner(n_trans, n_trans) * H_pos.transpose() + R_trans;
        S_trans = 0.5 * (S_trans + S_trans.transpose());
        auto S_ldlt = S_trans.ldlt();

        Eigen::MatrixXd K_trans = P.topLeftCorner(n_trans, n_trans) * H_pos.transpose()
            * S_ldlt.solve(Eigen::MatrixXd::Identity(d, d));

        next_mean = predicted.mean() + K_trans * trans_innov;
        next_rotation = R_icp;  // adopt ICP rotation directly

        Eigen::MatrixXd I_trans = Eigen::MatrixXd::Identity(n_trans, n_trans);
        Eigen::MatrixXd P_trans_updated = (I_trans - K_trans * H_pos) * P.topLeftCorner(n_trans, n_trans);

        // Build full augmented covariance with rotation from prior
        next_cov = predicted.covariance();
        next_cov.topLeftCorner(n_trans, n_trans) = 0.5 * (P_trans_updated + P_trans_updated.transpose());

        mahal = trans_innov.transpose() * S_ldlt.solve(trans_innov);
        log_det_S = S_ldlt.vectorD().array().abs().log().sum();
        innov_dim = d;
    } else {
        // Tracking: full augmented EKF with rotation innovation
        Eigen::VectorXd rot_innov(n_rot);
        if (d == 3) {
            Eigen::Matrix3d R_err = R_icp * predicted.rotation().transpose();
            rot_innov = Log_SO3(Eigen::Matrix3d(R_err));
        } else {
            Eigen::Matrix2d R_err = R_icp * predicted.rotation().transpose();
            rot_innov(0) = Log_SO2(Eigen::Matrix2d(R_err));
        }

        const int m_dim = d + n_rot;
        Eigen::VectorXd innovation(m_dim);
        innovation.head(d) = trans_innov;
        innovation.tail(n_rot) = rot_innov;

        Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
        H_aug.topLeftCorner(d, n_trans) = H_pos;
        H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::MatrixXd::Identity(n_rot, n_rot);

        const Eigen::MatrixXd& P = predicted.covariance();
        Eigen::MatrixXd S = H_aug * P * H_aug.transpose() + measurement_noise_;
        S = 0.5 * (S + S.transpose());

        auto S_ldlt = S.ldlt();
        Eigen::MatrixXd K = P * H_aug.transpose() * S_ldlt.solve(Eigen::MatrixXd::Identity(m_dim, m_dim));

        Eigen::VectorXd delta = K * innovation;
        next_mean = predicted.mean() + delta.head(n_trans);

        Eigen::VectorXd delta_rot = delta.tail(n_rot);
        if (d == 3) {
            next_rotation = Exp_SO3(Eigen::Vector3d(delta_rot)) * predicted.rotation();
        } else {
            next_rotation = Exp_SO2(delta_rot(0)) * Eigen::Matrix2d(predicted.rotation());
        }

        Eigen::MatrixXd I_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
        next_cov = (I_aug - K * H_aug) * P;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        mahal = innovation.transpose() * S_ldlt.solve(innovation);
        log_det_S = S_ldlt.vectorD().array().abs().log().sum();
        innov_dim = m_dim;
    }

    // Combined likelihood = L_template * L_pose
    double log_L_pose = -0.5 * (innov_dim * std::log(2.0 * M_PI) + log_det_S + mahal);

    // L_template: how well the template shape fits the measurement (from ICP)
    double log_L = icp_result.log_likelihood + log_L_pose;
    double likelihood = std::exp(std::clamp(log_L, -500.0, 500.0));
    if (!std::isfinite(likelihood) || likelihood <= 0.0) {
        likelihood = 1e-300;
    }

    return {
        models::TemplatePose(std::move(next_mean), std::move(next_cov),
                              std::move(next_rotation),
                              predicted.template_ptr(), predicted.pos_indices()),
        likelihood
    };
}

double TmEkf::gate(
    const Eigen::VectorXd& measurement,
    const models::TemplatePose& predicted) const {

    const int d = predicted.dim();
    const auto& pos_idx = predicted.pos_indices();

    // Centroid of measurement cloud
    const int M = static_cast<int>(measurement.size()) / d;
    Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
    Eigen::VectorXd centroid = meas_mat.rowwise().mean();

    // Predicted position
    Eigen::VectorXd t_pred = predicted.position();

    // Position-position covariance subblock
    Eigen::MatrixXd P_pos(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            P_pos(i, j) = predicted.covariance()(pos_idx[i], pos_idx[j]);
        }
    }

    Eigen::VectorXd nu = centroid - t_pred;
    return nu.transpose() * P_pos.ldlt().solve(nu);
}

} // namespace brew::filters
