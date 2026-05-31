#pragma once

#include "brew/core/filters/filter.hpp"
#include "brew/core/filters/ggiw_ekf.hpp"  // for detail::sqrtm_spd
#include "brew/core/models/trajectory.hpp"
#include "brew/core/models/ggiw_orientation.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace brew::filters {

/// EKF for Trajectory<GGIWOrientation<>> distributions.
/// Same trajectory predict/gate math as TrajectoryGGIWEKF; correct adds
/// eigenvector basis tracking that aligns new eigenvectors to the previous
/// basis after each update.
// @mex filter
// @mex_name TrajectoryGGIWOrientationEKF
// @mex_dist TrajectoryGGIWOrientation
// @mex_setters temporal_decay:scalar, forgetting_factor:scalar, scaling_parameter:scalar
template <int MaxWindow, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class TrajectoryGGIWOrientationEKF
    : public Filter<models::Trajectory<models::GGIWOrientation<Scalar, D, De>, MaxWindow>> {
public:
    using InnerDist = models::GGIWOrientation<Scalar, D, De>;
    using Dist = models::Trajectory<InnerDist, MaxWindow>;
    using TrajectoryType = Dist;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    TrajectoryGGIWOrientationEKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TrajectoryGGIWOrientationEKF<MaxWindow, Scalar, D, De>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->eta_ = eta_;
        c->tau_ = tau_;
        c->rho_ = rho_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        Dist result = prev;
        const int sd = prev.state_dim;

        // Gaussian (kinematic) prediction
        Eigen::VectorXd prev_last_state = prev.get_last_state();
        Eigen::VectorXd next_state = this->dyn_obj_->propagate_state(dt, prev_last_state);
        Eigen::MatrixXd F = this->dyn_obj_->get_state_mat(dt, prev_last_state);

        // Gamma prediction (measurement rate decay)
        const auto& prev_ggiw = prev.current();
        const double prev_alpha = prev_ggiw.alpha();
        const double prev_beta = prev_ggiw.beta();
        double next_alpha = prev_alpha / eta_;
        double next_beta = prev_beta / eta_;

        // IW prediction (extent decay, propagated through rotation if applicable)
        const double prev_v = prev_ggiw.v();
        const auto& prev_V = prev_ggiw.V();
        const int d = static_cast<int>(prev_V.rows());
        const double dof_floor = 2.0 * d + 2.0;
        double next_v = dof_floor + std::exp(-dt / tau_) * (prev_v - dof_floor);
        double scale = (next_v - dof_floor) / std::max(prev_v - dof_floor, 1e-12);
        Eigen::MatrixXd next_V = scale * this->dyn_obj_->propagate_extent(dt, prev_last_state, prev_V);

        // Advance ring buffer (slides if at cap); new tail slot zeroed
        result.advance_window();

        const int last = result.last_index();
        const int prev_last = last - 1;

        // Stacked covariance: fill new last row/col and auto-cov
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

        // Stacked mean + predicted GGIWOrientation marginal (basis preserved)
        result.mean_at(last) = next_state;
        InnerDist new_ggiw(
            next_alpha, next_beta,
            next_state,
            Eigen::MatrixXd(result.cov_at(last, last)),
            next_v, next_V);
        new_ggiw.basis() = prev_ggiw.basis();
        new_ggiw.eigenvalues() = prev_ggiw.eigenvalues();
        result.history_at(last) = std::move(new_ggiw);

        return result;
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const int sd = predicted.state_dim;
        const int last = predicted.last_index();
        const int live = predicted.stacked_size();

        const Eigen::VectorXd prev_state = predicted.get_last_state();

        const auto& pred_ggiw = predicted.current();
        const double prev_alpha = pred_ggiw.alpha();
        const double prev_beta = pred_ggiw.beta();
        const double prev_v = pred_ggiw.v();
        const auto& prev_V = pred_ggiw.V();
        const int d = static_cast<int>(prev_V.rows());

        // Measurement parsing: d×W point set; centroid & scatter
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
        Eigen::MatrixXd scatter = Eigen::MatrixXd::Zero(d, d);
        for (int j = 0; j < W; ++j) {
            const Eigen::VectorXd diff = Z.col(j) - mean_meas;
            scatter += diff * diff.transpose();
        }

        const double dof_floor = 2.0 * d + 2.0;
        Eigen::MatrixXd X_hat = prev_V / (prev_v - dof_floor);

        // Gaussian (kinematic) update on the live stacked slice
        Eigen::VectorXd est_meas = this->estimate_measurement(prev_state);
        Eigen::MatrixXd H = this->get_measurement_matrix(prev_state);

        const int m = static_cast<int>(H.rows());
        Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(m, live);
        H_dot.block(0, live - sd, m, sd) = H;

        Eigen::MatrixXd P_live = predicted.covariance().topLeftCorner(live, live);
        Eigen::VectorXd mean_live = predicted.mean().head(live);

        Eigen::VectorXd epsilon = mean_meas - est_meas;
        Eigen::MatrixXd N = epsilon * epsilon.transpose();

        Eigen::MatrixXd R_hat = rho_ * X_hat + this->measurement_noise_;
        Eigen::MatrixXd S = H_dot * P_live * H_dot.transpose()
                           + R_hat / static_cast<double>(W);

        Eigen::MatrixXd K = P_live * H_dot.transpose() * S.inverse();

        Eigen::VectorXd new_mean_live = mean_live + K * epsilon;
        Eigen::MatrixXd new_P_live = P_live - K * H_dot * P_live;

        // IW update: V += N_hat + scatter
        Eigen::MatrixXd sqrt_X = detail::sqrtm_spd(X_hat);
        Eigen::MatrixXd sqrt_R = detail::sqrtm_spd(R_hat);
        Eigen::MatrixXd sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd N_hat = sqrt_X * sqrt_R_inv * N * sqrt_R_inv.transpose() * sqrt_X.transpose();

        double next_v = prev_v + W;
        Eigen::MatrixXd next_V = prev_V + N_hat + scatter;

        // Gamma update: alpha += W, beta += 1
        double next_alpha = prev_alpha + W;
        double next_beta = prev_beta + 1.0;

        // Likelihood
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

        // Basis tracking: SVD corrected V, align columns/signs to previous basis
        // Greedy column matching on |U_new^T U_prev| to resolve the sign/ordering ambiguity
        // of the eigendecomposition and keep the rotation consistent across updates.
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(next_V, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U_new = svd.matrixU();
        Eigen::VectorXd S_vals = svd.singularValues();

        Eigen::MatrixXd U_prev = pred_ggiw.basis();
        if (U_prev.size() == 0) {
            U_prev = Eigen::MatrixXd::Identity(d, d);
        }

        Eigen::MatrixXd A = U_new.transpose() * U_prev;

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

        // Reconstruct V = U Σ U^T from the aligned basis.
        next_V = U_aligned * S_aligned.asDiagonal() * U_aligned.transpose();

        // Write corrected slice back; refresh history + aligned basis on last entry
        Dist result = predicted;
        result.mean().head(live) = new_mean_live;
        result.covariance().topLeftCorner(live, live) = new_P_live;

        for (int i = 0; i < predicted.window_size(); ++i) {
            result.history_at(i).mean() = result.mean_at(i);
        }
        auto& last_ggiw = result.history_at(last);
        last_ggiw.covariance() = Eigen::MatrixXd(result.cov_at(last, last));
        last_ggiw.alpha() = next_alpha;
        last_ggiw.beta() = next_beta;
        last_ggiw.v() = next_v;
        last_ggiw.V() = next_V;
        last_ggiw.basis() = U_aligned;
        last_ggiw.eigenvalues() = S_aligned.asDiagonal();

        result.commit_current_to_state_history();
        return { std::move(result), likelihood };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const Eigen::VectorXd state = predicted.get_last_state();
        const Eigen::MatrixXd P = predicted.get_last_cov();

        const auto& ggiw = predicted.current();
        const int d = static_cast<int>(ggiw.V().rows());
        const double dof_floor = 2.0 * d + 2.0;
        Eigen::MatrixXd X_hat = ggiw.V() / std::max(ggiw.v() - dof_floor, 1e-12);

        const Eigen::VectorXd z_hat = this->estimate_measurement(state);
        const Eigen::MatrixXd H = this->get_measurement_matrix(state);

        Eigen::VectorXd nu = measurement - z_hat;
        Eigen::MatrixXd S = H * P * H.transpose() + rho_ * X_hat + this->measurement_noise_;
        S = 0.5 * (S + S.transpose());

        return nu.transpose() * S.ldlt().solve(nu);
    }

    void set_temporal_decay(double eta) { eta_ = eta; }
    void set_forgetting_factor(double tau) { tau_ = tau; }
    void set_scaling_parameter(double rho) { rho_ = rho; }
    [[nodiscard]] static constexpr int window_size() { return Dist::max_window_size(); }

private:
    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;
};

} // namespace brew::filters
