#include "brew/filters/ggiw_orientation_ekf.hpp"
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

std::unique_ptr<Filter<models::GGIWOrientation>> GGIWOrientationEKF::clone() const {
    auto c = std::make_unique<GGIWOrientationEKF>();
    c->dyn_obj_ = dyn_obj_;
    c->h_ = h_;
    c->H_ = H_;
    c->process_noise_ = process_noise_;
    c->measurement_noise_ = measurement_noise_;
    c->eta_ = eta_;
    c->tau_ = tau_;
    c->rho_ = rho_;
    return c;
}

models::GGIWOrientation GGIWOrientationEKF::predict(
    double dt,
    const models::GGIWOrientation& prev) const {

    // Kinematic prediction
    const auto F = dyn_obj_->get_state_mat(dt, prev.mean());
    Eigen::VectorXd next_mean = dyn_obj_->propagate_state(dt, prev.mean());
    Eigen::MatrixXd next_cov = F * prev.covariance() * F.transpose() + process_noise_;

    // Gamma prediction (forgetting factor decay)
    double next_alpha = prev.alpha() / eta_;
    double next_beta = prev.beta() / eta_;

    // IW prediction (exponential dof decay + extent propagation)
    const int d = prev.extent_dim();
    const double dof_floor = 2.0 * d + 2.0;
    double next_v = dof_floor + std::exp(-dt / tau_) * (prev.v() - dof_floor);

    double scale = (next_v - dof_floor) / std::max(prev.v() - dof_floor, 1e-12);
    Eigen::MatrixXd next_V = scale * dyn_obj_->propagate_extent(dt, prev.mean(), prev.V());

    // Preserve basis through prediction
    models::GGIWOrientation result(
        std::move(next_mean), std::move(next_cov),
        next_alpha, next_beta, next_v, std::move(next_V));
    result.basis() = prev.basis();
    result.eigenvalues() = prev.eigenvalues();
    return result;
}

GGIWOrientationEKF::CorrectionResult GGIWOrientationEKF::correct(
    const Eigen::VectorXd& measurement,
    const models::GGIWOrientation& predicted) const {

    // ---- Standard GGIW EKF correction (same math as ggiw_ekf.cpp) ----
    const int d = predicted.extent_dim();

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
    Eigen::MatrixXd X_hat = predicted.V() / (predicted.v() - dof_floor);

    // Innovation
    const auto H = get_measurement_matrix(predicted.mean());
    const Eigen::VectorXd est_meas = estimate_measurement(predicted.mean());
    const Eigen::VectorXd epsilon = mean_meas - est_meas;

    const Eigen::MatrixXd N = epsilon * epsilon.transpose();

    // Measurement innovation covariance
    Eigen::MatrixXd R_hat = rho_ * X_hat + measurement_noise_;
    Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + R_hat / static_cast<double>(W);

    // Kalman gain
    Eigen::MatrixXd K = predicted.covariance() * H.transpose() * S.inverse();

    // N_hat: transformed innovation spread through extent
    Eigen::MatrixXd sqrt_X = sqrtm_spd(X_hat);
    Eigen::MatrixXd sqrt_R = sqrtm_spd(R_hat);
    Eigen::MatrixXd sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
    Eigen::MatrixXd N_hat = sqrt_X * sqrt_R_inv * N * sqrt_R_inv.transpose() * sqrt_X.transpose();

    // Update
    double next_alpha = predicted.alpha() + W;
    double next_beta = predicted.beta() + 1.0;
    Eigen::VectorXd next_mean = predicted.mean() + K * epsilon;
    Eigen::MatrixXd next_cov = predicted.covariance() - K * H * predicted.covariance();
    next_cov = 0.5 * (next_cov + next_cov.transpose());

    double next_v = predicted.v() + W;
    Eigen::MatrixXd next_V = predicted.V() + N_hat + scatter;

    // Log-likelihood (GGIW)
    const double v0 = predicted.v();
    const double v1 = next_v;
    const double a0 = predicted.alpha();
    const double a1 = next_alpha;
    const double b0 = predicted.beta();
    const double b1 = next_beta;

    double log_det_V0 = predicted.V().ldlt().vectorD().array().abs().log().sum();
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
        // For column k of U_prev, find the best matching column in U_new
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

        // Fix sign flip: negate column if correlation is negative
        if (A(best_col, k) < 0.0) {
            U_aligned.col(k) = -U_aligned.col(k);
        }
    }

    // Reconstruct V = U * S * U^T
    next_V = U_aligned * S_aligned.asDiagonal() * U_aligned.transpose();

    // Build result with updated basis
    models::GGIWOrientation result(
        std::move(next_mean), std::move(next_cov),
        next_alpha, next_beta, next_v, std::move(next_V));
    result.basis() = U_aligned;
    result.eigenvalues() = S_aligned.asDiagonal();

    return { std::move(result), std::exp(log_likelihood) };
}

double GGIWOrientationEKF::gate(
    const Eigen::VectorXd& measurement,
    const models::GGIWOrientation& predicted) const {

    const int d = predicted.extent_dim();
    const double dof_floor = 2.0 * d + 2.0;
    Eigen::MatrixXd X_hat = predicted.V() / std::max(predicted.v() - dof_floor, 1e-12);

    const auto H = get_measurement_matrix(predicted.mean());
    const Eigen::VectorXd z_hat = estimate_measurement(predicted.mean());
    const Eigen::VectorXd nu = measurement - z_hat;

    Eigen::MatrixXd S_gate = H * predicted.covariance() * H.transpose()
                       + rho_ * X_hat + measurement_noise_;
    S_gate = 0.5 * (S_gate + S_gate.transpose());

    return nu.transpose() * S_gate.ldlt().solve(nu);
}

} // namespace brew::filters
