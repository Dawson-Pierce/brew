#include "brew/filters/trajectory_gaussian_ekf.hpp"
#include <cmath>

namespace brew::filters {

using TrajectoryType = TrajectoryGaussianEKF::TrajectoryType;

std::unique_ptr<Filter<TrajectoryType>> TrajectoryGaussianEKF::clone() const {
    auto c = std::make_unique<TrajectoryGaussianEKF>();
    c->dyn_obj_ = dyn_obj_;
    c->h_ = h_;
    c->H_ = H_;
    c->process_noise_ = process_noise_;
    c->measurement_noise_ = measurement_noise_;
    c->l_window_ = l_window_;
    return c;
}

TrajectoryType TrajectoryGaussianEKF::predict(
    double dt,
    const TrajectoryType& prev) const {

    const int sd = prev.state_dim;
    const int ws = prev.window_size();

    // Determine windowing start index
    int start_idx = 0;
    if (ws >= l_window_) {
        start_idx = sd; // trim oldest state block
    }

    // Extract previous last state and windowed covariance
    Eigen::VectorXd prev_state = prev.get_last_state();
    const int total = static_cast<int>(prev.mean().size());
    Eigen::MatrixXd prev_cov = prev.covariance().block(
        start_idx, start_idx, total - start_idx, total - start_idx);

    // Dynamics
    Eigen::VectorXd next_state = dyn_obj_->propagate_state(dt, prev_state);
    Eigen::MatrixXd F = dyn_obj_->get_state_mat(dt, prev_state);

    // F_dot: block matrix that selects and transforms the last state
    const int cov_dim = static_cast<int>(prev_cov.rows());
    const int num_prev_blocks = cov_dim / sd;
    Eigen::MatrixXd F_dot = Eigen::MatrixXd::Zero(sd, cov_dim);
    F_dot.block(0, (num_prev_blocks - 1) * sd, sd, sd) = F;

    // Build new stacked covariance
    const int new_dim = cov_dim + sd;
    Eigen::MatrixXd new_cov(new_dim, new_dim);
    new_cov.topLeftCorner(cov_dim, cov_dim) = prev_cov;
    new_cov.topRightCorner(cov_dim, sd) = prev_cov * F_dot.transpose();
    new_cov.bottomLeftCorner(sd, cov_dim) = F_dot * prev_cov;
    new_cov.bottomRightCorner(sd, sd) = F_dot * prev_cov * F_dot.transpose() + process_noise_;

    // Build new stacked mean
    Eigen::VectorXd new_mean(cov_dim + sd);
    new_mean.head(cov_dim) = prev.mean().tail(total - start_idx);
    new_mean.tail(sd) = next_state;

    // Build result
    TrajectoryType result;
    result.state_dim = sd;
    result.mean() = new_mean;
    result.covariance() = new_cov;

    // Copy history from prev, append new marginal state
    result.history() = prev.history();
    result.history().push_back(models::Gaussian(
        next_state, new_cov.bottomRightCorner(sd, sd)));

    return result;
}

TrajectoryGaussianEKF::CorrectionResult TrajectoryGaussianEKF::correct(
    const Eigen::VectorXd& measurement,
    const TrajectoryType& predicted) const {

    const int sd = predicted.state_dim;
    const int ws = predicted.window_size();
    const Eigen::VectorXd prev_state = predicted.get_last_state();
    const Eigen::MatrixXd& prev_cov = predicted.covariance();

    // Measurement model
    Eigen::VectorXd est_meas = estimate_measurement(prev_state);
    Eigen::MatrixXd H = get_measurement_matrix(prev_state);

    // H_dot: measurement matrix that selects the last state block
    const int total_dim = static_cast<int>(prev_cov.rows());
    const int m = static_cast<int>(H.rows());
    Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(m, total_dim);
    H_dot.block(0, total_dim - sd, m, sd) = H;

    Eigen::VectorXd epsilon = measurement - est_meas;
    Eigen::MatrixXd S = H_dot * prev_cov * H_dot.transpose() + measurement_noise_;
    Eigen::MatrixXd K = prev_cov * H_dot.transpose() * S.inverse();

    Eigen::VectorXd new_mean = predicted.mean() + K * epsilon;
    Eigen::MatrixXd new_cov = prev_cov - K * H_dot * prev_cov;

    // Gaussian log-likelihood
    const int meas_dim = static_cast<int>(measurement.size());
    double log_det_S = S.ldlt().vectorD().array().log().sum();
    double mahal = epsilon.transpose() * S.ldlt().solve(epsilon);
    double log_likelihood = -0.5 * (meas_dim * std::log(2.0 * M_PI) + log_det_S + mahal);
    double likelihood = std::exp(log_likelihood);

    TrajectoryType result;
    result.state_dim = sd;
    result.mean() = new_mean;
    result.covariance() = new_cov;

    // Copy history and update windowed entries with corrected means/covs
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
    // Update last entry's covariance with corrected marginal
    result.history().back().covariance() =
        new_cov.block(total_dim - sd, total_dim - sd, sd, sd);

    return { std::move(result), likelihood };
}

double TrajectoryGaussianEKF::gate(
    const Eigen::VectorXd& measurement,
    const TrajectoryType& predicted) const {

    const Eigen::VectorXd state = predicted.get_last_state();
    const Eigen::MatrixXd P = predicted.get_last_cov();

    const Eigen::VectorXd z_hat = estimate_measurement(state);
    const Eigen::MatrixXd H = get_measurement_matrix(state);

    Eigen::VectorXd nu = measurement - z_hat;
    Eigen::MatrixXd S = H * P * H.transpose() + measurement_noise_;
    S = 0.5 * (S + S.transpose());

    return nu.transpose() * S.ldlt().solve(nu);
}

} // namespace brew::filters
