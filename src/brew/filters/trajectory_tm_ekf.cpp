#include "brew/filters/trajectory_tm_ekf.hpp"
#include <cmath>

namespace brew::filters {

// ---- Lie group helpers (same as tm_ekf.cpp) ----

static Eigen::Vector3d Log_SO3(const Eigen::Matrix3d& R) {
    const double cos_angle = (R.trace() - 1.0) / 2.0;
    const double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));
    if (angle < 1e-10) return Eigen::Vector3d::Zero();
    Eigen::Vector3d axis;
    axis << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    axis *= angle / (2.0 * std::sin(angle));
    return axis;
}

static Eigen::Matrix3d Exp_SO3(const Eigen::Vector3d& phi) {
    const double angle = phi.norm();
    if (angle < 1e-10) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = phi / angle;
    Eigen::Matrix3d K;
    K <<     0.0, -axis(2),  axis(1),
         axis(2),      0.0, -axis(0),
        -axis(1),  axis(0),      0.0;
    return Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * K * K;
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

using TrajectoryType = TrajectoryTmEkf::TrajectoryType;

std::unique_ptr<Filter<TrajectoryType>> TrajectoryTmEkf::clone() const {
    auto c = std::make_unique<TrajectoryTmEkf>();
    c->dyn_obj_ = dyn_obj_;
    c->h_ = h_;
    c->H_ = H_;
    c->process_noise_ = process_noise_;
    c->measurement_noise_ = measurement_noise_;
    c->rotation_process_noise_ = rotation_process_noise_;
    c->icp_ = icp_;
    c->l_window_ = l_window_;
    return c;
}

TrajectoryType TrajectoryTmEkf::predict(
    double dt,
    const TrajectoryType& prev) const {

    const int sd = prev.state_dim;  // translational state dim
    const int ws = prev.window_size();

    // Windowing
    int start_idx = 0;
    if (ws >= l_window_) {
        start_idx = sd;
    }

    Eigen::VectorXd prev_state = prev.get_last_state();
    const int total = static_cast<int>(prev.mean().size());
    Eigen::MatrixXd prev_cov = prev.covariance().block(
        start_idx, start_idx, total - start_idx, total - start_idx);

    // Dynamics
    Eigen::VectorXd next_state = dyn_obj_->propagate_state(dt, prev_state);
    Eigen::MatrixXd F = dyn_obj_->get_state_mat(dt, prev_state);
    Eigen::MatrixXd G = dyn_obj_->get_input_mat(dt, prev_state);

    // F_dot: extends stacked state
    const int cov_dim = static_cast<int>(prev_cov.rows());
    const int num_prev_blocks = cov_dim / sd;
    Eigen::MatrixXd F_dot = Eigen::MatrixXd::Zero(sd, cov_dim);
    F_dot.block(0, (num_prev_blocks - 1) * sd, sd, sd) = F;

    const int new_dim = cov_dim + sd;
    Eigen::MatrixXd new_cov(new_dim, new_dim);
    new_cov.topLeftCorner(cov_dim, cov_dim) = prev_cov;
    new_cov.topRightCorner(cov_dim, sd) = prev_cov * F_dot.transpose();
    new_cov.bottomLeftCorner(sd, cov_dim) = F_dot * prev_cov;
    new_cov.bottomRightCorner(sd, sd) =
        F_dot * prev_cov * F_dot.transpose() + G * process_noise_ * G.transpose();

    Eigen::VectorXd new_mean(cov_dim + sd);
    new_mean.head(cov_dim) = prev.mean().tail(total - start_idx);
    new_mean.tail(sd) = next_state;

    // Rotation: propagate via dynamics (identity for SingleIntegrator)
    const auto& prev_tp = prev.current();
    Eigen::MatrixXd next_rotation = dyn_obj_->propagate_extent(dt, prev_state, prev_tp.rotation());

    // Augmented covariance for the current step's TemplatePose
    const int n_rot = prev_tp.rot_dim();
    const int n_aug = sd + n_rot;
    Eigen::MatrixXd aug_cov = Eigen::MatrixXd::Zero(n_aug, n_aug);
    aug_cov.topLeftCorner(sd, sd) = new_cov.bottomRightCorner(sd, sd);
    // Rotation covariance: propagate from previous + process noise
    const Eigen::MatrixXd& prev_aug = prev_tp.covariance();
    aug_cov.bottomRightCorner(n_rot, n_rot) =
        prev_aug.bottomRightCorner(n_rot, n_rot) + rotation_process_noise_;

    // Build result
    TrajectoryType result;
    result.state_dim = sd;
    result.mean() = new_mean;
    result.covariance() = new_cov;

    // Copy history, append new TemplatePose marginal
    result.history() = prev.history();
    result.history().push_back(models::TemplatePose(
        next_state, aug_cov, next_rotation,
        prev_tp.template_ptr(), prev_tp.pos_indices()));

    return result;
}

TrajectoryTmEkf::CorrectionResult TrajectoryTmEkf::correct(
    const Eigen::VectorXd& measurement,
    const TrajectoryType& predicted) const {

    const int sd = predicted.state_dim;
    const Eigen::VectorXd prev_state = predicted.get_last_state();
    const Eigen::MatrixXd& prev_cov = predicted.covariance();
    const int total_dim = static_cast<int>(prev_cov.rows());

    const auto& pred_tp = predicted.current();
    const int d = pred_tp.dim();
    const int n_rot = pred_tp.rot_dim();
    const auto& pos_idx = pred_tp.pos_indices();

    // Reconstruct measurement as PointCloud (d×M, column-major)
    const int M = static_cast<int>(measurement.size()) / d;
    Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
    template_matching::PointCloud meas_cloud(meas_mat);

    // Current pose estimate
    Eigen::VectorXd t_pred = pred_tp.position();

    // Run ICP (embed to 3D if 2D)
    template_matching::IcpResult icp_result;
    Eigen::VectorXd t_icp(d);
    Eigen::MatrixXd R_icp(d, d);

    if (d == 2) {
        Eigen::Matrix3d R_pred_3d = Eigen::Matrix3d::Identity();
        R_pred_3d.topLeftCorner(2, 2) = pred_tp.rotation();
        Eigen::Vector3d t_pred_3d = Eigen::Vector3d::Zero();
        t_pred_3d.head(2) = t_pred;

        const auto& templ_pts = pred_tp.get_template().points();
        Eigen::MatrixXd templ_3d = Eigen::MatrixXd::Zero(3, templ_pts.cols());
        templ_3d.topRows(2) = templ_pts;
        template_matching::PointCloud templ_cloud_3d(templ_3d);

        Eigen::MatrixXd meas_3d = Eigen::MatrixXd::Zero(3, M);
        meas_3d.topRows(2) = meas_mat;
        template_matching::PointCloud meas_cloud_3d(meas_3d);

        icp_result = icp_->align(templ_cloud_3d, meas_cloud_3d, R_pred_3d, t_pred_3d);

        R_icp = icp_result.rotation.topLeftCorner(2, 2);
        t_icp = icp_result.translation.head(2);
    } else {
        icp_result = icp_->align(
            pred_tp.get_template(), meas_cloud,
            Eigen::Matrix3d(pred_tp.rotation()),
            Eigen::Vector3d(t_pred));
        R_icp = icp_result.rotation;
        t_icp = icp_result.translation;
    }

    // --- Coupled augmented correction on current step (like regular TmEkf) ---

    // H_pos: selects position from translational state
    Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, sd);
    for (int i = 0; i < d; ++i) {
        H_pos(i, pos_idx[i]) = 1.0;
    }

    // Translation innovation
    Eigen::VectorXd trans_innov = t_icp - H_pos * prev_state;

    // Rotation innovation
    Eigen::VectorXd rot_innov(n_rot);
    if (d == 3) {
        Eigen::Matrix3d R_err = R_icp * pred_tp.rotation().transpose();
        rot_innov = Log_SO3(Eigen::Matrix3d(R_err));
    } else {
        Eigen::Matrix2d R_err = R_icp * Eigen::Matrix2d(pred_tp.rotation().transpose());
        rot_innov(0) = Log_SO2(Eigen::Matrix2d(R_err));
    }

    // Full augmented innovation
    const int m_dim = d + n_rot;
    Eigen::VectorXd innovation(m_dim);
    innovation.head(d) = trans_innov;
    innovation.tail(n_rot) = rot_innov;

    // Augmented measurement Jacobian for current step
    const int n_aug = sd + n_rot;
    Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
    H_aug.topLeftCorner(d, sd) = H_pos;
    H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::MatrixXd::Identity(n_rot, n_rot);

    // Current step's augmented covariance
    const Eigen::MatrixXd& P_aug = pred_tp.covariance();

    // Innovation covariance (using full augmented state)
    const Eigen::MatrixXd& R_meas = measurement_noise_;
    Eigen::MatrixXd S = H_aug * P_aug * H_aug.transpose() + R_meas;
    S = 0.5 * (S + S.transpose());

    // Augmented Kalman gain
    Eigen::MatrixXd K_aug = P_aug * H_aug.transpose() * S.inverse();

    // Augmented correction delta
    Eigen::VectorXd delta = K_aug * innovation;
    Eigen::VectorXd delta_trans = delta.head(sd);
    Eigen::VectorXd delta_rot = delta.tail(n_rot);

    // Rotation update (on manifold)
    Eigen::MatrixXd next_rotation(d, d);
    if (d == 3) {
        next_rotation = Exp_SO3(Eigen::Vector3d(delta_rot)) * pred_tp.rotation();
    } else {
        next_rotation = Exp_SO2(delta_rot(0)) * Eigen::Matrix2d(pred_tp.rotation());
    }

    // Updated augmented covariance
    Eigen::MatrixXd I_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
    Eigen::MatrixXd next_aug_cov = (I_aug - K_aug * H_aug) * P_aug;
    next_aug_cov = 0.5 * (next_aug_cov + next_aug_cov.transpose());

    // --- Stacked translational correction ---
    // Use the translational innovation to correct the full stacked state

    Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(d, total_dim);
    H_dot.block(0, total_dim - sd, d, sd) = H_pos;

    Eigen::MatrixXd S_trans = H_dot * prev_cov * H_dot.transpose()
                            + R_meas.topLeftCorner(d, d);
    Eigen::MatrixXd K_stacked = prev_cov * H_dot.transpose() * S_trans.inverse();
    Eigen::VectorXd new_mean = predicted.mean() + K_stacked * trans_innov;
    Eigen::MatrixXd new_cov = prev_cov - K_stacked * H_dot * prev_cov;

    // --- Likelihood ---

    double log_det_S = S.ldlt().vectorD().array().abs().log().sum();
    double mahal = innovation.transpose() * S.ldlt().solve(innovation);
    double log_L_pose = -0.5 * (m_dim * std::log(2.0 * M_PI) + log_det_S + mahal);

    double log_L = icp_result.log_likelihood + log_L_pose;
    double likelihood = std::exp(std::clamp(log_L, -500.0, 500.0));
    if (!std::isfinite(likelihood) || likelihood <= 0.0) {
        likelihood = 1e-300;
    }

    // --- Build result ---

    Eigen::VectorXd next_trans_mean = new_mean.tail(sd);

    TrajectoryType result;
    result.state_dim = sd;
    result.mean() = new_mean;
    result.covariance() = new_cov;

    // Copy history and update windowed entries with corrected means
    result.history() = predicted.history();
    const int new_steps = static_cast<int>(new_mean.size()) / sd;
    Eigen::MatrixXd rearranged = Eigen::Map<const Eigen::MatrixXd>(
        new_mean.data(), sd, new_steps);
    const int h = static_cast<int>(result.history().size());
    for (int i = 0; i < new_steps && i < h; ++i) {
        int hist_idx = h - new_steps + i;
        if (hist_idx >= 0) {
            result.history()[hist_idx].mean() = rearranged.col(i);
        }
    }

    // Update last entry with corrected pose
    auto& last = result.history().back();
    last.mean() = next_trans_mean;
    last.covariance() = next_aug_cov;
    last.rotation() = next_rotation;

    return { std::move(result), likelihood };
}

double TrajectoryTmEkf::gate(
    const Eigen::VectorXd& measurement,
    const models::Trajectory<models::TemplatePose>& predicted) const {

    const auto& tp = predicted.current();
    const int d = tp.dim();
    const auto& pos_idx = tp.pos_indices();

    // Centroid of measurement cloud
    const int M = static_cast<int>(measurement.size()) / d;
    Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
    Eigen::VectorXd centroid = meas_mat.rowwise().mean();

    // Predicted position
    Eigen::VectorXd t_pred = tp.position();

    // Position-position covariance subblock from current step's augmented cov
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

} // namespace brew::filters
