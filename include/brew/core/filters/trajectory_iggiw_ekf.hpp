#pragma once

#include "brew/core/filters/filter.hpp"
#include "brew/core/models/trajectory.hpp"
#include "brew/core/models/iggiw.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace brew::filters {

/// EKF for Trajectory<IGGIW<>> distributions.
///
/// Same intensity-and-extent math as IGGIWEKF (Inverse-Gamma on per-cell
/// intensity, Inverse-Wishart on extent, Gaussian on kinematics), but
/// maintains a stacked window of past kinematic states so the trajectory
/// is smoothed by each correction. Measurement format and hyperparameter
/// semantics match IGGIWEKF exactly, including the centroid_power knob
/// that controls how aggressively hot spots pull the location estimate.
// @mex filter
// @mex_name TrajectoryIGGIWEKF
// @mex_dist TrajectoryIGGIW
// @mex_setters eta:scalar, lambda:scalar, omega:scalar, intensity_forgetting_factor:scalar, intensity_growth:scalar, extent_forgetting_factor:scalar, centroid_power:scalar
template <int MaxWindow, typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class TrajectoryIGGIWEKF
    : public Filter<models::Trajectory<models::IGGIW<Scalar, D, De>, MaxWindow>> {
public:
    using InnerDist = models::IGGIW<Scalar, D, De>;
    using Dist = models::Trajectory<InnerDist, MaxWindow>;
    using TrajectoryType = Dist;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    TrajectoryIGGIWEKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TrajectoryIGGIWEKF<MaxWindow, Scalar, D, De>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->eta_ = eta_;
        c->lambda_ = lambda_;
        c->omega_ = omega_;
        c->delta_gamma_ = delta_gamma_;
        c->gamma_growth_ = gamma_growth_;
        c->delta_extent_ = delta_extent_;
        c->centroid_power_ = centroid_power_;
        return c;
    }

    [[nodiscard]] Dist predict(double dt, const Dist& prev) const override {
        Dist result = prev;
        const int sd = prev.state_dim;
        (void)sd;

        const Eigen::VectorXd prev_last_state = prev.get_last_state();
        const Eigen::VectorXd next_state = this->dyn_obj_->propagate_state(dt, prev_last_state);
        const Eigen::MatrixXd F = this->dyn_obj_->get_state_mat(dt, prev_last_state);

        // ---- IG predict on the marginal intensity Inverse-Gamma ----
        const auto& prev_iggiw = prev.current();
        const double eps = std::numeric_limits<double>::epsilon();

        const double alpha_prev = std::max(prev_iggiw.alpha(), 1.0 + eps);
        const double beta_prev  = std::max(prev_iggiw.beta(),  eps);
        const double gamma_bar  = gamma_growth_ * beta_prev / (alpha_prev - 1.0);

        double next_alpha = std::max(1.0 + delta_gamma_ * (alpha_prev - 1.0), 1.0 + eps);
        double next_beta  = std::max(gamma_bar * (next_alpha - 1.0), eps);

        // ---- IW predict on the extent ----
        const int d = prev_iggiw.extent_dim();
        const double dof_floor = 2.0 * d + 2.0;
        const double v_prev = std::max(prev_iggiw.v(), dof_floor + eps);

        Eigen::MatrixXd X_bar = prev_iggiw.V() / (v_prev - dof_floor);
        X_bar = this->dyn_obj_->propagate_extent(dt, prev_last_state, X_bar);
        X_bar = 0.5 * (X_bar + X_bar.transpose());

        double next_v = std::max(dof_floor + delta_extent_ * (v_prev - dof_floor),
                                 dof_floor + eps);
        Eigen::MatrixXd next_V = X_bar * (next_v - dof_floor);
        next_V = 0.5 * (next_V + next_V.transpose());

        // ---- Advance ring buffer and fill the new stacked slot ----
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
                + this->process_noise_;
        } else {
            result.cov_at(last, last) = this->process_noise_;
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

        const auto& pred_iggiw = predicted.current();
        const int d = pred_iggiw.extent_dim();

        // ---- Unpack packed [pos_d; weight_1] x N measurement ----
        Eigen::MatrixXd positions;
        Eigen::VectorXd weights;
        unpack_measurement(measurement, d, positions, weights);

        for (int i = 0; i < weights.size(); ++i) {
            if (!(weights(i) > 0.0) || !std::isfinite(weights(i))) {
                throw std::invalid_argument(
                    "TrajectoryIGGIWEKF measurement weights must be positive and finite.");
            }
        }

        const int n = static_cast<int>(weights.size());
        const double eps = std::numeric_limits<double>::epsilon();
        const double sum_w = std::max(weights.sum(), eps);

        // Intensity summary uses the generalized power mean
        //   r = ( (1/N) * sum(w_j^p) )^(1/p)
        // so centroid_power biases both location and intensity together.
        // r = mean(w) at p = 1 and -> max(w) as p grows.
        double r;
        if (centroid_power_ == 1.0) {
            r = std::max(weights.mean(), eps);
        } else {
            const double mean_wp =
                weights.array().pow(centroid_power_).mean();
            r = std::pow(std::max(mean_wp, eps), 1.0 / centroid_power_);
            r = std::max(r, eps);
        }

        // ---- Centroid: w^p tunes hot-spot bias (p == 1 -> linear) ----
        const Eigen::VectorXd weights_p =
            (centroid_power_ == 1.0)
                ? weights
                : Eigen::VectorXd(weights.array().pow(centroid_power_));
        const double sum_wp = std::max(weights_p.sum(), eps);
        const Eigen::VectorXd z_bar = (positions * weights_p) / sum_wp;

        // ---- Z_meas keeps linear weights so extent reflects full spread ----
        Eigen::MatrixXd Z_meas = Eigen::MatrixXd::Zero(d, d);
        for (int j = 0; j < n; ++j) {
            const Eigen::VectorXd dz = positions.col(j) - z_bar;
            Z_meas += weights(j) * dz * dz.transpose();
        }
        Z_meas /= sum_w;
        Z_meas = 0.5 * (Z_meas + Z_meas.transpose());

        // ---- Expected extent ----
        const double dof_floor = 2.0 * d + 2.0;
        const double v_safe = std::max(pred_iggiw.v(), dof_floor + eps);
        Eigen::MatrixXd X_hat = pred_iggiw.V() / (v_safe - dof_floor);
        X_hat = 0.5 * (X_hat + X_hat.transpose());

        // ---- Stacked Kalman update on the live window slice ----
        const Eigen::VectorXd prev_state = predicted.get_last_state();
        const Eigen::VectorXd z_hat = this->estimate_measurement(prev_state);
        const Eigen::MatrixXd H = this->get_measurement_matrix(prev_state);

        const int m = static_cast<int>(H.rows());
        Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(m, live);
        H_dot.block(0, live - sd, m, sd) = H;

        Eigen::MatrixXd P_live = predicted.covariance().topLeftCorner(live, live);
        Eigen::VectorXd mean_live = predicted.mean().head(live);

        const Eigen::VectorXd innovation = z_bar - z_hat;

        Eigen::MatrixXd R_hat = X_hat / lambda_ + this->measurement_noise_;
        R_hat = 0.5 * (R_hat + R_hat.transpose());

        Eigen::MatrixXd S = H_dot * P_live * H_dot.transpose() + R_hat;
        S = 0.5 * (S + S.transpose());

        Eigen::MatrixXd K = P_live * H_dot.transpose()
            * S.ldlt().solve(Eigen::MatrixXd::Identity(m, m));

        Eigen::VectorXd new_mean_live = mean_live + K * innovation;
        Eigen::MatrixXd new_P_live = P_live - K * H_dot * P_live;
        new_P_live = 0.5 * (new_P_live + new_P_live.transpose());

        // ---- IG update on intensity ----
        const double next_alpha = pred_iggiw.alpha() + eta_;
        const double next_beta  = pred_iggiw.beta()  + eta_ * r;

        // ---- IW update on extent ----
        const double next_v = pred_iggiw.v() + omega_;
        Eigen::MatrixXd next_V = pred_iggiw.V() + omega_ * Z_meas;
        next_V = 0.5 * (next_V + next_V.transpose());

        // ---- Likelihood (same decomposition as IGGIWEKF) ----
        const double log_likelihood =
            log_intensity_marginal(r, pred_iggiw.alpha(), pred_iggiw.beta())
          + log_gaussian(z_bar, z_hat, S)
          + log_wishart_extent(Z_meas, X_hat, omega_);
        const double likelihood = std::exp(log_likelihood);

        // ---- Write back into the stacked trajectory ----
        Dist result = predicted;
        result.mean().head(live) = new_mean_live;
        result.covariance().topLeftCorner(live, live) = new_P_live;

        for (int i = 0; i < predicted.window_size(); ++i) {
            result.history_at(i).mean() = result.mean_at(i);
        }
        auto& last_iggiw = result.history_at(last);
        last_iggiw.covariance() = Eigen::MatrixXd(result.cov_at(last, last));
        last_iggiw.alpha() = next_alpha;
        last_iggiw.beta()  = next_beta;
        last_iggiw.v()     = next_v;
        last_iggiw.V()     = next_V;

        return { std::move(result), likelihood };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const auto& pred_iggiw = predicted.current();
        const int d = pred_iggiw.extent_dim();

        Eigen::MatrixXd positions;
        Eigen::VectorXd weights;
        unpack_measurement(measurement, d, positions, weights);

        const double eps = std::numeric_limits<double>::epsilon();
        const Eigen::VectorXd weights_p =
            (centroid_power_ == 1.0)
                ? weights
                : Eigen::VectorXd(weights.array().pow(centroid_power_));
        const double sum_wp = std::max(weights_p.sum(), eps);
        const Eigen::VectorXd z_bar = (positions * weights_p) / sum_wp;

        const double dof_floor = 2.0 * d + 2.0;
        const double v_safe = std::max(pred_iggiw.v(), dof_floor + eps);
        Eigen::MatrixXd X_hat = pred_iggiw.V() / (v_safe - dof_floor);
        X_hat = 0.5 * (X_hat + X_hat.transpose());

        const Eigen::VectorXd state = predicted.get_last_state();
        const Eigen::MatrixXd P = predicted.get_last_cov();
        const Eigen::VectorXd z_hat = this->estimate_measurement(state);
        const Eigen::MatrixXd H = this->get_measurement_matrix(state);

        const Eigen::VectorXd nu = z_bar - z_hat;
        Eigen::MatrixXd R_hat = X_hat / lambda_ + this->measurement_noise_;
        Eigen::MatrixXd S = H * P * H.transpose() + R_hat;
        S = 0.5 * (S + S.transpose());

        return nu.transpose() * S.ldlt().solve(nu);
    }

    // ---- Setters (validated, mirror IGGIWEKF) ----

    void set_eta(double eta) {
        if (!(eta > 0.0) || !std::isfinite(eta))
            throw std::invalid_argument("eta must be positive and finite.");
        eta_ = eta;
    }
    void set_lambda(double lambda) {
        if (!(lambda > 0.0) || !std::isfinite(lambda))
            throw std::invalid_argument("lambda must be positive and finite.");
        lambda_ = lambda;
    }
    void set_omega(double omega) {
        if (!(omega > 0.0) || !std::isfinite(omega))
            throw std::invalid_argument("omega must be positive and finite.");
        omega_ = omega;
    }
    void set_intensity_forgetting_factor(double delta_gamma) {
        if (!(delta_gamma > 0.0 && delta_gamma <= 1.0) || !std::isfinite(delta_gamma))
            throw std::invalid_argument("delta_gamma must be in (0, 1].");
        delta_gamma_ = delta_gamma;
    }
    void set_intensity_growth(double gamma_growth) {
        if (!(gamma_growth > 0.0) || !std::isfinite(gamma_growth))
            throw std::invalid_argument("gamma_growth must be positive and finite.");
        gamma_growth_ = gamma_growth;
    }
    void set_extent_forgetting_factor(double delta_extent) {
        if (!(delta_extent > 0.0 && delta_extent <= 1.0) || !std::isfinite(delta_extent))
            throw std::invalid_argument("delta_extent must be in (0, 1].");
        delta_extent_ = delta_extent;
    }
    void set_centroid_power(double centroid_power) {
        if (!(centroid_power > 0.0) || !std::isfinite(centroid_power))
            throw std::invalid_argument("centroid_power must be positive and finite.");
        centroid_power_ = centroid_power;
    }

    [[nodiscard]] static constexpr int window_size() { return Dist::max_window_size(); }

private:
    // Packed [pos_d; weight_1] x N format, matching IGGIWEKF::unpack_measurement.
    static void unpack_measurement(
        const Eigen::VectorXd& measurement, int d,
        Eigen::MatrixXd& positions, Eigen::VectorXd& weights) {
        const int stride = d + 1;
        if (measurement.size() >= stride && measurement.size() % stride == 0) {
            const int n = static_cast<int>(measurement.size()) / stride;
            Eigen::Map<const Eigen::MatrixXd> packed(measurement.data(), stride, n);
            positions = packed.topRows(d);
            weights = packed.row(d).transpose();
            return;
        }
        if (measurement.size() == d) {
            positions = measurement;
            weights = Eigen::VectorXd::Ones(1);
            return;
        }
        throw std::invalid_argument(
            "TrajectoryIGGIWEKF measurement must be length d or packed length (d+1)*N.");
    }

    // ---- Log-likelihood helpers (mirror IGGIWEKF) ----

    static double logdet_spd(const Eigen::MatrixXd& A) {
        Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
        const auto diag = ldlt.vectorD().array();
        return diag.abs().max(std::numeric_limits<double>::min()).log().sum();
    }

    static double trace_solve_spd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
        return ldlt.solve(B).trace();
    }

    static double log_multivariate_gamma(double x, int d) {
        double out = static_cast<double>(d) * (static_cast<double>(d) - 1.0)
                   * 0.25 * std::log(M_PI);
        for (int i = 1; i <= d; ++i) {
            out += std::lgamma(x + 0.5 * (1.0 - static_cast<double>(i)));
        }
        return out;
    }

    double log_intensity_marginal(double r, double alpha, double beta) const {
        const double eps = std::numeric_limits<double>::epsilon();
        r = std::max(r, eps);
        alpha = std::max(alpha, eps);
        beta = std::max(beta, eps);
        return eta_ * std::log(eta_)
             + (eta_ - 1.0) * std::log(r)
             - std::lgamma(eta_)
             + alpha * std::log(beta)
             + std::lgamma(alpha + eta_)
             - std::lgamma(alpha)
             - (alpha + eta_) * std::log(beta + eta_ * r);
    }

    static double log_gaussian(
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance) {
        const int d = static_cast<int>(z.size());
        const Eigen::VectorXd innovation = z - mean;
        const double log_det = logdet_spd(covariance);
        const double quad = innovation.transpose() * covariance.ldlt().solve(innovation);
        return -0.5 * (
            static_cast<double>(d) * std::log(2.0 * M_PI)
            + log_det + quad
        );
    }

    static double log_wishart_extent(
        const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, double omega) {
        const int d = static_cast<int>(Z.rows());
        const double eps = 1e-12;
        Eigen::MatrixXd Z_reg = 0.5 * (Z + Z.transpose());
        Eigen::MatrixXd X_reg = 0.5 * (X + X.transpose());
        Z_reg += eps * Eigen::MatrixXd::Identity(d, d);
        X_reg += eps * Eigen::MatrixXd::Identity(d, d);
        const Eigen::MatrixXd Sigma = X_reg / omega;
        const double log_det_Z = logdet_spd(Z_reg);
        const double log_det_Sigma = logdet_spd(Sigma);
        const double tr_term = trace_solve_spd(Sigma, Z_reg);
        return 0.5 * (omega - static_cast<double>(d) - 1.0) * log_det_Z
             - 0.5 * tr_term
             - 0.5 * omega * static_cast<double>(d) * std::log(2.0)
             - 0.5 * omega * log_det_Sigma
             - log_multivariate_gamma(0.5 * omega, d);
    }

    double eta_ = 4.0;
    double lambda_ = 4.0;
    double omega_ = 5.0;
    double delta_gamma_ = 0.9;
    double gamma_growth_ = 1.0;
    double delta_extent_ = 0.9;
    double centroid_power_ = 1.0;
};

} // namespace brew::filters
