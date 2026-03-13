#include "brew/filters/tm_ekf.hpp"
#include <cmath>

namespace brew::filters {

// ---- Lie group helpers ----

static Eigen::Vector3d Log_SO3(const Eigen::Matrix3d& R) {
    const double cos_angle = (R.trace() - 1.0) / 2.0;
    const double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));

    if (angle < 1e-10) {
        return Eigen::Vector3d::Zero();
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
    return c;
}

models::TemplatePose TmEkf::predict(
    double dt,
    const models::TemplatePose& prev) const {

    const int n_trans = prev.trans_dim();
    const int n_rot = prev.rot_dim();
    const int n_aug = prev.aug_dim();

    // Propagate translational state via dynamics
    const auto F = dyn_obj_->get_state_mat(dt, prev.mean());
    const auto G = dyn_obj_->get_input_mat(dt, prev.mean());
    Eigen::VectorXd next_mean = dyn_obj_->propagate_state(dt, prev.mean());

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

    // Rotation: use propagate_extent if dynamics provides it, else hold constant
    Eigen::MatrixXd next_rotation = dyn_obj_->propagate_extent(dt, prev.mean(), prev.rotation());

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

    // Current pose estimate
    Eigen::VectorXd t_pred = predicted.position();

    // Run ICP (embed to 3D if 2D)
    template_matching::IcpResult icp_result;
    Eigen::VectorXd t_icp(d);
    Eigen::MatrixXd R_icp(d, d);

    if (d == 2) {
        // Embed 2D → 3D
        Eigen::Matrix3d R_pred_3d = Eigen::Matrix3d::Identity();
        R_pred_3d.topLeftCorner(2, 2) = predicted.rotation();
        Eigen::Vector3d t_pred_3d = Eigen::Vector3d::Zero();
        t_pred_3d.head(2) = t_pred;

        // Embed template
        const auto& templ_pts = predicted.get_template().points();
        Eigen::MatrixXd templ_3d = Eigen::MatrixXd::Zero(3, templ_pts.cols());
        templ_3d.topRows(2) = templ_pts;
        template_matching::PointCloud templ_cloud_3d(templ_3d);

        // Embed measurement
        Eigen::MatrixXd meas_3d = Eigen::MatrixXd::Zero(3, M);
        meas_3d.topRows(2) = meas_mat;
        template_matching::PointCloud meas_cloud_3d(meas_3d);

        icp_result = icp_->align(templ_cloud_3d, meas_cloud_3d, R_pred_3d, t_pred_3d);

        // Project back to 2D
        R_icp = icp_result.rotation.topLeftCorner(2, 2);
        t_icp = icp_result.translation.head(2);
    } else {
        icp_result = icp_->align(
            predicted.get_template(), meas_cloud,
            Eigen::Matrix3d(predicted.rotation()),
            Eigen::Vector3d(t_pred));

        R_icp = icp_result.rotation;
        t_icp = icp_result.translation;
    }

    // Build H_pos: selects position indices from translational state
    Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, n_trans);
    for (int i = 0; i < d; ++i) {
        H_pos(i, pos_idx[i]) = 1.0;
    }

    // Translation innovation
    Eigen::VectorXd trans_innov = t_icp - H_pos * predicted.mean();

    // Rotation innovation
    Eigen::VectorXd rot_innov(n_rot);
    if (d == 3) {
        Eigen::Matrix3d R_err = R_icp * predicted.rotation().transpose();
        rot_innov = Log_SO3(Eigen::Matrix3d(R_err));
    } else {
        Eigen::Matrix2d R_err = R_icp * predicted.rotation().transpose();
        rot_innov(0) = Log_SO2(Eigen::Matrix2d(R_err));
    }

    // Full augmented innovation
    const int m_dim = d + n_rot;
    Eigen::VectorXd innovation(m_dim);
    innovation.head(d) = trans_innov;
    innovation.tail(n_rot) = rot_innov;

    // Augmented measurement Jacobian
    Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
    H_aug.topLeftCorner(d, n_trans) = H_pos;
    H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::MatrixXd::Identity(n_rot, n_rot);

    // Innovation covariance
    const Eigen::MatrixXd& P = predicted.covariance();
    Eigen::MatrixXd S = H_aug * P * H_aug.transpose() + measurement_noise_;
    S = 0.5 * (S + S.transpose());

    // Kalman gain
    Eigen::MatrixXd K = P * H_aug.transpose() * S.inverse();

    // State correction
    Eigen::VectorXd delta = K * innovation;

    // Translational update (Euclidean)
    Eigen::VectorXd next_mean = predicted.mean() + delta.head(n_trans);

    // Rotation update (on manifold)
    Eigen::VectorXd delta_rot = delta.tail(n_rot);
    Eigen::MatrixXd next_rotation(d, d);
    if (d == 3) {
        next_rotation = Exp_SO3(Eigen::Vector3d(delta_rot)) * predicted.rotation();
    } else {
        next_rotation = Exp_SO2(delta_rot(0)) * Eigen::Matrix2d(predicted.rotation());
    }

    // Covariance update
    Eigen::MatrixXd I_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
    Eigen::MatrixXd next_cov = (I_aug - K * H_aug) * P;
    next_cov = 0.5 * (next_cov + next_cov.transpose());

    // Likelihood = L_template * L_pose
    // L_pose: Gaussian innovation likelihood
    double log_det_S = S.ldlt().vectorD().array().abs().log().sum();
    double mahal = innovation.transpose() * S.ldlt().solve(innovation);
    double log_L_pose = -0.5 * (m_dim * std::log(2.0 * M_PI) + log_det_S + mahal);

    // L_template: from ICP result
    double likelihood = icp_result.likelihood * std::exp(log_L_pose);

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
