#include "brew/filters/trajectory_ggiw_orientation_ekf.hpp"
#include <cmath>
#include <algorithm>

namespace brew::filters {

// Symmetric positive-definite matrix square root via eigendecomposition
static Eigen::MatrixXd sqrtm_spd(const Eigen::MatrixXd& M) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
    Eigen::VectorXd d = es.eigenvalues();
    for (int i = 0; i < d.size(); ++i) {
        d(i) = std::sqrt(std::max(d(i), std::numeric_limits<double>::epsilon()));
    }
    return es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
}

std::unique_ptr<Filter<models::TrajectoryGGIWOrientation>> TrajectoryGGIWOrientationEKF::clone() const {
    auto c = std::make_unique<TrajectoryGGIWOrientationEKF>();
    c->dyn_obj_ = dyn_obj_;
    c->h_ = h_;
    c->H_ = H_;
    c->process_noise_ = process_noise_;
    c->measurement_noise_ = measurement_noise_;
    c->eta_ = eta_;
    c->tau_ = tau_;
    c->rho_ = rho_;
    c->l_window_ = l_window_;
    return c;
}

models::TrajectoryGGIWOrientation TrajectoryGGIWOrientationEKF::predict(
    double dt,
    const models::TrajectoryGGIWOrientation& prev) const {

    const int sd = prev.state_dim;
    const int ws = prev.window_size;

    // Windowing
    int start_idx = 0;
    if (ws >= l_window_) {
        start_idx = sd;
    }

    Eigen::VectorXd prev_state = prev.get_last_state();
    const int total = static_cast<int>(prev.mean().size());
    Eigen::MatrixXd prev_cov = prev.covariance().block(
        start_idx, start_idx, total - start_idx, total - start_idx);

    // Kinematic trajectory prediction
    Eigen::VectorXd next_state = dyn_obj_->propagate_state(dt, prev_state);
    Eigen::MatrixXd F = dyn_obj_->get_state_mat(dt, prev_state);

    const int cov_dim = static_cast<int>(prev_cov.rows());
    const int num_prev_blocks = cov_dim / sd;
    Eigen::MatrixXd F_dot = Eigen::MatrixXd::Zero(sd, cov_dim);
    F_dot.block(0, (num_prev_blocks - 1) * sd, sd, sd) = F;

    const int new_dim = cov_dim + sd;
    Eigen::MatrixXd new_cov(new_dim, new_dim);
    new_cov.topLeftCorner(cov_dim, cov_dim) = prev_cov;
    new_cov.topRightCorner(cov_dim, sd) = prev_cov * F_dot.transpose();
    new_cov.bottomLeftCorner(sd, cov_dim) = F_dot * prev_cov;
    new_cov.bottomRightCorner(sd, sd) = F_dot * prev_cov * F_dot.transpose() + process_noise_;

    Eigen::VectorXd new_mean(cov_dim + sd);
    new_mean.head(cov_dim) = prev.mean().tail(total - start_idx);
    new_mean.tail(sd) = next_state;

    // GGIW gamma/IW prediction
    const double prev_alpha = prev.alpha();
    const double prev_beta = prev.beta();
    const double prev_v = prev.v();
    const Eigen::MatrixXd& prev_V = prev.V();
    const int d = static_cast<int>(prev_V.rows());

    double next_alpha = prev_alpha / eta_;
    double next_beta = prev_beta / eta_;

    const double dof_floor = 2.0 * d + 2.0;
    double next_v = dof_floor + std::exp(-dt / tau_) * (prev_v - dof_floor);
    double scale = (next_v - dof_floor) / std::max(prev_v - dof_floor, 1e-12);
    Eigen::MatrixXd next_V = scale * dyn_obj_->propagate_extent(dt, prev_state, prev_V);

    // Build result
    models::TrajectoryGGIWOrientation result;
    result.state_dim = sd;
    result.init_idx = prev.init_idx;
    result.window_size = ws + 1;
    result.mean() = new_mean;
    result.covariance() = new_cov;
    result.alpha() = next_alpha;
    result.beta() = next_beta;
    result.v() = next_v;
    result.V() = next_V;

    // Preserve basis through prediction
    result.basis() = prev.basis();
    result.eigenvalues() = prev.eigenvalues();

    // Update histories
    result.cov_history() = prev.cov_history();
    result.cov_history().push_back(new_cov.bottomRightCorner(sd, sd));

    result.alpha_history() = prev.alpha_history();
    result.alpha_history().push_back(next_alpha);
    result.beta_history() = prev.beta_history();
    result.beta_history().push_back(next_beta);
    result.v_history() = prev.v_history();
    result.v_history().push_back(next_v);
    result.V_history() = prev.V_history();
    result.V_history().push_back(next_V);

    // Update mean_history
    if (ws < l_window_) {
        result.mean_history() = result.rearrange_states();
    } else {
        const int hist_cols = static_cast<int>(prev.mean_history().cols());
        const int new_steps = static_cast<int>(new_mean.size()) / sd;
        Eigen::MatrixXd new_hist(sd, hist_cols + 1);
        new_hist.leftCols(hist_cols - l_window_ + 2) = prev.mean_history().leftCols(hist_cols - l_window_ + 2);
        Eigen::MatrixXd rearranged = Eigen::Map<const Eigen::MatrixXd>(new_mean.data(), sd, new_steps);
        new_hist.rightCols(new_steps) = rearranged;
        result.mean_history() = new_hist;
    }

    return result;
}

TrajectoryGGIWOrientationEKF::CorrectionResult TrajectoryGGIWOrientationEKF::correct(
    const Eigen::VectorXd& measurement,
    const models::TrajectoryGGIWOrientation& predicted) const {

    const int sd = predicted.state_dim;
    const int ws = predicted.window_size;
    const Eigen::VectorXd prev_state = predicted.get_last_state();
    const Eigen::MatrixXd& prev_cov = predicted.covariance();
    const int total_dim = static_cast<int>(prev_cov.rows());

    const double prev_alpha = predicted.alpha();
    const double prev_beta = predicted.beta();
    const double prev_v = predicted.v();
    const Eigen::MatrixXd& prev_V = predicted.V();
    const int d = static_cast<int>(prev_V.rows());

    // Determine W (number of measurements)
    int W;
    Eigen::MatrixXd Z;
    if (measurement.size() > d && measurement.size() % d == 0) {
        W = static_cast<int>(measurement.size()) / d;
        Z = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, W);
    } else {
        W = 1;
        Z = measurement;
    }

    const Eigen::VectorXd mean_meas = Z.rowwise().mean();

    // Scatter matrix
    Eigen::MatrixXd scatter = Eigen::MatrixXd::Zero(d, d);
    for (int j = 0; j < W; ++j) {
        const Eigen::VectorXd diff = Z.col(j) - mean_meas;
        scatter += diff * diff.transpose();
    }

    // Expected extent
    const double dof_floor = 2.0 * d + 2.0;
    Eigen::MatrixXd X_hat = prev_V / (prev_v - dof_floor);

    // Measurement model
    Eigen::VectorXd est_meas = estimate_measurement(prev_state);
    Eigen::MatrixXd H = get_measurement_matrix(prev_state);

    // H_dot
    const int m = static_cast<int>(H.rows());
    Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(m, total_dim);
    H_dot.block(0, total_dim - sd, m, sd) = H;

    Eigen::VectorXd epsilon = mean_meas - est_meas;
    Eigen::MatrixXd N = epsilon * epsilon.transpose();

    Eigen::MatrixXd R_hat = rho_ * X_hat + measurement_noise_;
    Eigen::MatrixXd S = H_dot * prev_cov * H_dot.transpose()
                       + R_hat / static_cast<double>(W);

    Eigen::MatrixXd K = prev_cov * H_dot.transpose() * S.inverse();

    // N_hat
    Eigen::MatrixXd sqrt_X = sqrtm_spd(X_hat);
    Eigen::MatrixXd sqrt_R = sqrtm_spd(R_hat);
    Eigen::MatrixXd sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
    Eigen::MatrixXd N_hat = sqrt_X * sqrt_R_inv * N * sqrt_R_inv.transpose() * sqrt_X.transpose();

    // Update
    Eigen::VectorXd new_mean = predicted.mean() + K * epsilon;
    Eigen::MatrixXd new_cov = prev_cov - K * H_dot * prev_cov;

    double next_alpha = prev_alpha + W;
    double next_beta = prev_beta + 1.0;
    double next_v = prev_v + W;
    Eigen::MatrixXd next_V = prev_V + N_hat + scatter;

    // Log-likelihood
    const double v0 = prev_v;
    const double v1 = next_v;
    const double a0 = prev_alpha;
    const double a1 = next_alpha;
    const double b0 = prev_beta;
    const double b1 = next_beta;

    double log_det_V0 = prev_V.ldlt().vectorD().array().abs().log().sum();
    double log_det_V1 = next_V.ldlt().vectorD().array().abs().log().sum();
    double log_det_Xhat = X_hat.ldlt().vectorD().array().abs().log().sum();
    double log_det_S = S.ldlt().vectorD().array().abs().log().sum();

    double gammaln_sum = 0.0;
    for (int i = 0; i < d; ++i) {
        gammaln_sum += std::lgamma((v1 - d - 1.0) / 2.0 + (1.0 - (i + 1)) / 2.0);
        gammaln_sum -= std::lgamma((v0 - d - 1.0) / 2.0 + (1.0 - (i + 1)) / 2.0);
    }

    double log_likelihood =
        (v0 - d - 1.0) / 2.0 * log_det_V0
      - (v1 - d - 1.0) / 2.0 * log_det_V1
      + gammaln_sum
      + 0.5 * log_det_Xhat
      - 0.5 * log_det_S
      + std::lgamma(a1) - std::lgamma(a0)
      + a0 * std::log(b0) - a1 * std::log(b1)
      - (W * std::log(M_PI) + std::log(static_cast<double>(W))) * d / 2.0;

    double likelihood = std::exp(log_likelihood);

    // ---- Basis tracking: SVD of corrected V ----
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(next_V, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U_new = svd.matrixU();
    Eigen::VectorXd S_vals = svd.singularValues();

    // Previous basis (defaults to identity if not yet set)
    Eigen::MatrixXd U_prev = predicted.basis();
    if (U_prev.size() == 0) {
        U_prev = Eigen::MatrixXd::Identity(d, d);
    }

    // Alignment: A = U_new^T * U_prev
    Eigen::MatrixXd A = U_new.transpose() * U_prev;

    // Permute columns of U_new to maximize diagonal of A (greedy matching)
    Eigen::MatrixXd U_aligned = Eigen::MatrixXd::Zero(d, d);
    Eigen::VectorXd S_aligned = Eigen::VectorXd::Zero(d);
    std::vector<bool> used(d, false);

    for (int k = 0; k < d; ++k) {
        double best_val = -1.0;
        int best_col = 0;
        for (int j = 0; j < d; ++j) {
            if (used[j]) continue;
            double val = std::abs(A(j, k));
            if (val > best_val) {
                best_val = val;
                best_col = j;
            }
        }
        used[best_col] = true;
        U_aligned.col(k) = U_new.col(best_col);
        S_aligned(k) = S_vals(best_col);

        // Fix sign flip
        if (A(best_col, k) < 0.0) {
            U_aligned.col(k) = -U_aligned.col(k);
        }
    }

    // Reconstruct V = U * S * U^T
    next_V = U_aligned * S_aligned.asDiagonal() * U_aligned.transpose();

    // Build result
    models::TrajectoryGGIWOrientation result;
    result.state_dim = sd;
    result.init_idx = predicted.init_idx;
    result.window_size = ws;
    result.mean() = new_mean;
    result.covariance() = new_cov;
    result.alpha() = next_alpha;
    result.beta() = next_beta;
    result.v() = next_v;
    result.V() = next_V;

    // Store aligned basis and eigenvalues
    result.basis() = U_aligned;
    result.eigenvalues() = S_aligned.asDiagonal();

    // Update histories
    result.cov_history() = predicted.cov_history();
    if (!result.cov_history().empty()) {
        result.cov_history().back() = new_cov.block(total_dim - sd, total_dim - sd, sd, sd);
    }

    result.alpha_history() = predicted.alpha_history();
    if (!result.alpha_history().empty()) result.alpha_history().back() = next_alpha;
    result.beta_history() = predicted.beta_history();
    if (!result.beta_history().empty()) result.beta_history().back() = next_beta;
    result.v_history() = predicted.v_history();
    if (!result.v_history().empty()) result.v_history().back() = next_v;
    result.V_history() = predicted.V_history();
    if (!result.V_history().empty()) result.V_history().back() = next_V;

    // Update mean_history
    const int new_steps = static_cast<int>(new_mean.size()) / sd;
    Eigen::MatrixXd rearranged = Eigen::Map<const Eigen::MatrixXd>(new_mean.data(), sd, new_steps);
    if (predicted.mean_history().cols() > 0) {
        Eigen::MatrixXd new_hist = predicted.mean_history();
        const int hist_cols = static_cast<int>(new_hist.cols());
        new_hist.rightCols(std::min(new_steps, hist_cols)) =
            rearranged.rightCols(std::min(new_steps, hist_cols));
        result.mean_history() = new_hist;
    } else {
        result.mean_history() = rearranged;
    }

    return { std::move(result), likelihood };
}

double TrajectoryGGIWOrientationEKF::gate(
    const Eigen::VectorXd& measurement,
    const models::TrajectoryGGIWOrientation& predicted) const {

    const Eigen::VectorXd state = predicted.get_last_state();
    const Eigen::MatrixXd P = predicted.get_last_cov();

    const int d = static_cast<int>(predicted.V().rows());
    const double dof_floor = 2.0 * d + 2.0;
    Eigen::MatrixXd X_hat = predicted.V() / std::max(predicted.v() - dof_floor, 1e-12);

    const Eigen::VectorXd z_hat = estimate_measurement(state);
    const Eigen::MatrixXd H = get_measurement_matrix(state);

    Eigen::VectorXd nu = measurement - z_hat;
    Eigen::MatrixXd S = H * P * H.transpose() + rho_ * X_hat + measurement_noise_;
    S = 0.5 * (S + S.transpose());

    return nu.transpose() * S.ldlt().solve(nu);
}

} // namespace brew::filters
