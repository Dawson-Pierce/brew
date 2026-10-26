#pragma once

#include "brew/shared/filter_traits.hpp"
#include "brew/shared/parameter_validation.hpp"

#include "brew/shared/filter_base.hpp"
#include "brew/ggiw/filters/ggiw_ekf.hpp"
#include "brew/trajectory_ggiw/trajectory_ggiw_model.hpp"
#include "brew/ggiw/ggiw_model.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace brew::filters {

// @mex filter
// @mex_name TrajectoryGGIWEKF
// @mex_dist TrajectoryGGIW
// @mex_setters temporal_decay:scalar, forgetting_factor:scalar, scaling_parameter:scalar, window_size:int, ci_weight:scalar

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class TrajectoryGGIWEKF
    : public Filter<models::TrajectoryGGIW<Scalar, D, De>> {
public:
    using InnerDist = models::GGIW<Scalar, D, De>;
    using Dist = models::TrajectoryGGIW<Scalar, D, De>;
    using TrajectoryType = Dist;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    TrajectoryGGIWEKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TrajectoryGGIWEKF<Scalar, D, De>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->eta_ = eta_;
        c->tau_ = tau_;
        c->rho_ = rho_;
        c->window_size_ = window_size_;
        c->ci_weight_ = ci_weight_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        Dist result = prev;
        const int sd = prev.state_dim;

        Eigen::VectorXd prev_last_state = prev.get_last_state();
        Eigen::VectorXd next_state = this->dyn_obj_->propagate_state(dt, prev_last_state);
        Eigen::MatrixXd F = this->dyn_obj_->get_state_mat(dt, prev_last_state);
        const Eigen::MatrixXd& Q = this->process_noise_;

        const auto& prev_ggiw = prev.current();
        const double prev_alpha = prev_ggiw.alpha();
        const double prev_beta = prev_ggiw.beta();
        double next_alpha = prev_alpha / eta_;
        double next_beta = prev_beta / eta_;

        const double prev_v = prev_ggiw.v();
        const auto& prev_V = prev_ggiw.V();
        const int d = static_cast<int>(prev_V.rows());
        const double dof_floor = 2.0 * d + 2.0;
        double next_v = dof_floor + std::exp(-dt / tau_) * (prev_v - dof_floor);
        double scale = (next_v - dof_floor) / std::max(prev_v - dof_floor, 1e-12);
        Eigen::MatrixXd next_V = scale * this->dyn_obj_->propagate_extent(dt, prev_last_state, prev_V);

        result.advance_window();

        const int last = result.last_index();
        const int prev_last = last - 1;

        if (prev_last >= 0) {
            for (int k = 0; k < last; ++k) {
                result.cov_at(last, k) = F * result.cov_at(prev_last, k);
                result.cov_at(k, last) = result.cov_at(last, k).transpose();
            }
            result.cov_at(last, last) =
                F * result.cov_at(prev_last, prev_last) * F.transpose()
                + Q;
        } else {
            result.cov_at(last, last) = Q;
        }

        result.mean_at(last) = next_state;
        result.history_at(last) = InnerDist(
            next_alpha, next_beta,
            next_state,
            Eigen::MatrixXd(result.cov_at(last, last)),
            next_v, next_V);

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

        if (measurement.size() < d || measurement.size() % d != 0) {
            throw std::invalid_argument(
                "TrajectoryGGIWEKF measurement must contain one or more packed d-dimensional points.");
        }

        int W;
        Eigen::MatrixXd Z;
        if (measurement.size() > d) {
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
        Eigen::MatrixXd X_hat = prev_V / std::max(prev_v - dof_floor, 1e-12);

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

        const Eigen::MatrixXd Sinv = S.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd K = P_live * H_dot.transpose() * Sinv;

        Eigen::VectorXd new_mean_live = mean_live + K * epsilon;
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(live, live);
        const Eigen::MatrixXd A = I - K * H_dot;
        Eigen::MatrixXd new_P_live = A * P_live * A.transpose()
            + K * (R_hat / static_cast<double>(W)) * K.transpose();
        new_P_live = 0.5 * (new_P_live + new_P_live.transpose());

        Eigen::MatrixXd sqrt_X = detail::sqrtm_spd(X_hat);
        Eigen::MatrixXd sqrt_R = detail::sqrtm_spd(R_hat);
        Eigen::MatrixXd sqrt_S = detail::sqrtm_spd(S);
        Eigen::MatrixXd sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd sqrt_S_inv = sqrt_S.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd N_hat = sqrt_X * sqrt_S_inv * N * sqrt_S_inv.transpose() * sqrt_X.transpose();
        Eigen::MatrixXd Z_hat = sqrt_X * sqrt_R_inv * scatter
                              * sqrt_R_inv.transpose() * sqrt_X.transpose();

        double next_v = prev_v + W;
        Eigen::MatrixXd next_V = prev_V + N_hat + Z_hat;

        double next_alpha = prev_alpha + W;
        double next_beta = prev_beta + 1.0;

        const double v0 = prev_v;
        const double v1 = next_v;
        const double a0 = prev_alpha;
        const double a1 = next_alpha;
        const double b0 = prev_beta;
        const double b1 = next_beta;

        const double log_det_V0 = detail::logdet_spd(prev_V);
        const double log_det_V1 = detail::logdet_spd(next_V);
        const double log_det_Xhat = detail::logdet_spd(X_hat);
        const double log_det_Rhat = detail::logdet_spd(R_hat);
        const double log_det_S = detail::logdet_spd(S);

        double gammaln_sum = 0.0;
        for (int i = 0; i < d; ++i) {
            gammaln_sum += std::lgamma((v1 - d - 1.0) / 2.0 + (1.0 - (i + 1)) / 2.0);
            gammaln_sum -= std::lgamma((v0 - d - 1.0) / 2.0 + (1.0 - (i + 1)) / 2.0);
        }

        double log_likelihood =
            (v0 - d - 1.0) / 2.0 * log_det_V0
          - (v1 - d - 1.0) / 2.0 * log_det_V1
          + gammaln_sum
          + 0.5 * W * log_det_Xhat
          - 0.5 * (W - 1) * log_det_Rhat
          - 0.5 * log_det_S
          + std::lgamma(a1) - std::lgamma(a0)
          + a0 * std::log(b0) - a1 * std::log(b1)
          - (W * std::log(M_PI) + std::log(static_cast<double>(W))) * d / 2.0;

        double likelihood = detail::safe_exp(log_likelihood);

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

    // Track-to-track covariance-intersection update.  ``z`` is a persistent network estimate, not
    // an independent raw measurement, so its GGIW hyperparameters must not be added to the ground
    // trajectory.  CI remains consistent under unknown cross-correlation.  The current trajectory
    // marginal is CI-fused with z, then the full live window is reconstructed while preserving the
    // ground trajectory's conditional distribution of past states given its current state.
    [[nodiscard]] CorrectionResult correct_component(const InnerDist& z, const Dist& predicted) const {
        const int sd = predicted.state_dim;
        const int live = predicted.stacked_size();
        const auto& pred_ggiw = predicted.current();
        Eigen::MatrixXd P_live = predicted.covariance().topLeftCorner(live, live);
        Eigen::VectorXd mean_live = predicted.mean().head(live);
        const int current = live - sd;
        const Eigen::VectorXd mg = mean_live.tail(sd);
        Eigen::MatrixXd Pg = P_live.bottomRightCorner(sd, sd);
        Eigen::MatrixXd Pz = z.covariance();
        Pg = 0.5 * (Pg + Pg.transpose());
        Pz = 0.5 * (Pz + Pz.transpose());
        if (z.mean().size() != sd || Pz.rows() != sd || Pz.cols() != sd)
            return { predicted, 0.0 };

        const double omega = std::clamp(ci_weight_, 0.0, 1.0);
        const Eigen::MatrixXd Jg = Pg.inverse();
        const Eigen::MatrixXd Jz = Pz.inverse();
        Eigen::MatrixXd Jf = omega * Jg + (1.0 - omega) * Jz;
        Jf = 0.5 * (Jf + Jf.transpose());
        Eigen::LDLT<Eigen::MatrixXd> jf_ldlt(Jf);
        if (jf_ldlt.info() != Eigen::Success || !jf_ldlt.isPositive())
            return { predicted, 0.0 };
        const Eigen::MatrixXd Pf = jf_ldlt.solve(Eigen::MatrixXd::Identity(sd, sd));
        const Eigen::VectorXd mf = Pf * (omega * Jg * mg + (1.0 - omega) * Jz * z.mean());

        Eigen::VectorXd new_mean_live = mean_live;
        Eigen::MatrixXd new_P_live = P_live;
        if (current > 0) {
            const Eigen::MatrixXd Ppc = P_live.block(0, current, current, sd);
            const Eigen::MatrixXd A = Ppc * Jg;
            new_mean_live.head(current) += A * (mf - mg);
            new_P_live.topLeftCorner(current, current) += A * (Pf - Pg) * A.transpose();
            new_P_live.block(0, current, current, sd) = A * Pf;
            new_P_live.block(current, 0, sd, current) = Pf * A.transpose();
        }
        new_mean_live.tail(sd) = mf;
        new_P_live.bottomRightCorner(sd, sd) = Pf;
        new_P_live = 0.5 * (new_P_live + new_P_live.transpose());

        const double next_alpha = omega * pred_ggiw.alpha() + (1.0 - omega) * z.alpha();
        const double next_beta  = omega * pred_ggiw.beta()  + (1.0 - omega) * z.beta();
        const double next_v     = omega * pred_ggiw.v()     + (1.0 - omega) * z.v();
        Eigen::MatrixXd next_V  = omega * pred_ggiw.V()     + (1.0 - omega) * z.V();

        Eigen::MatrixXd S = Pg + Pz;
        S = 0.5 * (S + S.transpose());
        Eigen::LDLT<Eigen::MatrixXd> s_ldlt(S);
        if (s_ldlt.info() != Eigen::Success || !s_ldlt.isPositive())
            return { predicted, 0.0 };
        const Eigen::VectorXd epsilon = z.mean() - mg;
        const double log_det_S = s_ldlt.vectorD().array().log().sum();
        const double maha = epsilon.dot(s_ldlt.solve(epsilon));
        const double log_like = -0.5 * (maha + log_det_S + sd * std::log(2.0 * M_PI));

        Dist result = predicted;
        result.mean().head(live) = new_mean_live;
        result.covariance().topLeftCorner(live, live) = new_P_live;
        for (int i = 0; i < predicted.window_size(); ++i)
            result.history_at(i).mean() = result.mean_at(i);
        const int last = predicted.last_index();
        auto& last_ggiw = result.history_at(last);
        last_ggiw.covariance() = Eigen::MatrixXd(result.cov_at(last, last));
        last_ggiw.alpha() = next_alpha;
        last_ggiw.beta()  = next_beta;
        last_ggiw.v()     = next_v;
        last_ggiw.V()     = next_V;
        return { std::move(result), std::exp(log_like) };
    }

    [[nodiscard]] double gate_component(const InnerDist& z, const Dist& predicted) const {
        const Eigen::VectorXd state = predicted.get_last_state();
        const Eigen::MatrixXd P = predicted.get_last_cov();
        const Eigen::MatrixXd Hin = this->get_measurement_matrix(state);
        const Eigen::VectorXd nu = Hin * z.mean() - this->estimate_measurement(state);
        Eigen::MatrixXd R = Hin * z.covariance() * Hin.transpose() + this->measurement_noise_;
        Eigen::MatrixXd S = Hin * P * Hin.transpose() + R;
        S = 0.5 * (S + S.transpose());
        return nu.transpose() * S.ldlt().solve(nu);
    }

    void set_temporal_decay(double eta) { eta_ = brew::detail::positive_finite(eta, "Temporal decay"); }
    void set_forgetting_factor(double tau) { tau_ = brew::detail::positive_finite(tau, "Forgetting time"); }
    void set_scaling_parameter(double rho) { rho_ = brew::detail::positive_finite(rho, "Extent scale"); }
    void set_window_size(int n) { window_size_ = (n > 0 ? n : 1); }
    void set_ci_weight(double omega) { ci_weight_ = brew::detail::unit_interval(omega, "CI weight"); }
    [[nodiscard]] int window_size() const { return window_size_; }

private:
    int window_size_ = Dist::kDefaultWindow;

private:
    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;
    double ci_weight_ = 0.5;
};

}

namespace brew::filters {

template <typename Scalar, int D, int De>
struct default_filter<models::TrajectoryGGIW<Scalar, D, De>> { using type = TrajectoryGGIWEKF<Scalar, D, De>; };
}
