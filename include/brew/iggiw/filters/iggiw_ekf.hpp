#pragma once
#include "brew/shared/filter_traits.hpp"

#include "brew/shared/filter_base.hpp"
#include "brew/iggiw/iggiw_model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace brew::filters {

/// Extended Kalman Filter for IGGIW distributions.
// @mex filter
// @mex_name IGGIWEKF
// @mex_dist IGGIW
// @mex_setters eta:scalar, lambda:scalar, omega:scalar, intensity_forgetting_factor:scalar, intensity_growth:scalar, extent_forgetting_factor:scalar, centroid_power:scalar
template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class IGGIWEKF : public Filter<models::IGGIW<Scalar, D, De>> {
public:
    using Dist = models::IGGIW<Scalar, D, De>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    IGGIWEKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<IGGIWEKF<Scalar, D, De>>();
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
        const auto F = this->dyn_obj_->get_state_mat(dt, prev.mean());

        Eigen::VectorXd next_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
        Eigen::MatrixXd next_cov = F * prev.covariance() * F.transpose() + this->process_noise_;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        const double eps = std::numeric_limits<double>::epsilon();

        const double alpha_prev = std::max(prev.alpha(), 1.0 + eps);
        const double beta_prev = std::max(prev.beta(), eps);
        const double gamma_bar = gamma_growth_ * beta_prev / (alpha_prev - 1.0);

        double next_alpha = 1.0 + delta_gamma_ * (alpha_prev - 1.0);
        next_alpha = std::max(next_alpha, 1.0 + eps);

        double next_beta = gamma_bar * (next_alpha - 1.0);
        next_beta = std::max(next_beta, eps);

        const int d = prev.extent_dim();
        const double dof_floor = static_cast<double>(2 * d + 2);
        const double v_prev = std::max(prev.v(), dof_floor + eps);

        Eigen::MatrixXd X_bar = prev.V() / (v_prev - dof_floor);
        X_bar = this->dyn_obj_->propagate_extent(dt, prev.mean(), X_bar);
        X_bar = 0.5 * (X_bar + X_bar.transpose());

        double next_v = dof_floor + delta_extent_ * (v_prev - dof_floor);
        next_v = std::max(next_v, dof_floor + eps);

        Eigen::MatrixXd next_V = X_bar * (next_v - dof_floor);
        next_V = 0.5 * (next_V + next_V.transpose());

        return Dist(
            next_alpha, next_beta,
            std::move(next_mean), std::move(next_cov),
            next_v, std::move(next_V));
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const int d = predicted.extent_dim();

        Eigen::MatrixXd positions;
        Eigen::VectorXd weights;
        unpack_measurement(measurement, d, positions, weights);

        for (int i = 0; i < weights.size(); ++i) {
            if (!(weights(i) > 0.0) || !std::isfinite(weights(i))) {
                throw std::invalid_argument("IGGIWEKF measurement weights must be positive and finite.");
            }
        }

        const int n = static_cast<int>(weights.size());
        const double eps = std::numeric_limits<double>::epsilon();
        const double sum_w = std::max(weights.sum(), eps);

        // Intensity summary statistic uses the generalized power mean
        //   r = ( (1/N) * sum(w_j^p) )^(1/p)
        // so the same centroid_power knob biases the InverseGamma update
        // toward hot spots. r = mean(w) at p = 1, and r -> max(w) as p
        // increases — pulling beta/(alpha-1) from "average reflectivity"
        // toward "peak reflectivity" in lock-step with the centroid.
        double r;
        if (centroid_power_ == 1.0) {
            r = std::max(weights.mean(), eps);
        } else {
            const double mean_wp =
                weights.array().pow(centroid_power_).mean();
            r = std::pow(std::max(mean_wp, eps), 1.0 / centroid_power_);
            r = std::max(r, eps);
        }

        // Centroid uses w^p instead of w so the user can dial how much
        // the per-cell intensity dominates the location average:
        //   p = 1  -> standard intensity-weighted centroid (default)
        //   p > 1  -> centroid pulled harder toward hot spots
        //   p < 1  -> centroid closer to the geometric (unweighted) mean
        const Eigen::VectorXd weights_p =
            (centroid_power_ == 1.0)
                ? weights
                : Eigen::VectorXd(weights.array().pow(centroid_power_));
        const double sum_wp = std::max(weights_p.sum(), eps);
        const Eigen::VectorXd z_bar = (positions * weights_p) / sum_wp;

        // Z_meas keeps the linear-weight spread around z_bar so the extent
        // estimate still reflects the full intensity-weighted distribution
        // of cells (only the centroid is sharpened by p).
        Eigen::MatrixXd Z_meas = Eigen::MatrixXd::Zero(d, d);
        for (int j = 0; j < n; ++j) {
            const Eigen::VectorXd dz = positions.col(j) - z_bar;
            Z_meas += weights(j) * dz * dz.transpose();
        }
        Z_meas /= sum_w;
        Z_meas = 0.5 * (Z_meas + Z_meas.transpose());

        const double dof_floor = static_cast<double>(2 * d + 2);
        const double v_safe = std::max(predicted.v(), dof_floor + eps);

        Eigen::MatrixXd X_hat = predicted.V() / (v_safe - dof_floor);
        X_hat = 0.5 * (X_hat + X_hat.transpose());

        const auto H = this->get_measurement_matrix(predicted.mean());
        const Eigen::VectorXd z_hat = this->estimate_measurement(predicted.mean());
        const Eigen::VectorXd innovation = z_bar - z_hat;

        Eigen::MatrixXd R_hat = X_hat / lambda_ + this->measurement_noise_;
        R_hat = 0.5 * (R_hat + R_hat.transpose());

        Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + R_hat;
        S = 0.5 * (S + S.transpose());

        Eigen::MatrixXd K = predicted.covariance() * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(d, d));

        double next_alpha = predicted.alpha() + eta_;
        double next_beta = predicted.beta() + eta_ * r;

        Eigen::VectorXd next_mean = predicted.mean() + K * innovation;
        Eigen::MatrixXd next_cov = predicted.covariance() - K * S * K.transpose();
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        double next_v = predicted.v() + omega_;
        Eigen::MatrixXd next_V = predicted.V() + omega_ * Z_meas;
        next_V = 0.5 * (next_V + next_V.transpose());

        const double log_likelihood =
            log_intensity_marginal(r, predicted.alpha(), predicted.beta())
          + log_gaussian(z_bar, z_hat, S)
          + log_wishart_extent(Z_meas, X_hat, omega_);

        return {
            Dist(next_alpha, next_beta,
                 std::move(next_mean), std::move(next_cov),
                 next_v, std::move(next_V)),
            std::exp(log_likelihood)
        };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const int d = predicted.extent_dim();

        Eigen::MatrixXd positions;
        Eigen::VectorXd weights;
        unpack_measurement(measurement, d, positions, weights);

        const double eps = std::numeric_limits<double>::epsilon();

        // Match the w^p centroid used in correct() so the gate compares
        // against the same point estimate the update will move toward.
        const Eigen::VectorXd weights_p =
            (centroid_power_ == 1.0)
                ? weights
                : Eigen::VectorXd(weights.array().pow(centroid_power_));
        const double sum_wp = std::max(weights_p.sum(), eps);
        const Eigen::VectorXd z_bar = (positions * weights_p) / sum_wp;

        const double dof_floor = static_cast<double>(2 * d + 2);
        const double v_safe = std::max(predicted.v(), dof_floor + eps);

        Eigen::MatrixXd X_hat = predicted.V() / (v_safe - dof_floor);
        X_hat = 0.5 * (X_hat + X_hat.transpose());

        const auto H = this->get_measurement_matrix(predicted.mean());
        const Eigen::VectorXd z_hat = this->estimate_measurement(predicted.mean());
        const Eigen::VectorXd innovation = z_bar - z_hat;

        Eigen::MatrixXd R_hat = X_hat / lambda_ + this->measurement_noise_;
        R_hat = 0.5 * (R_hat + R_hat.transpose());

        Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + R_hat;
        S = 0.5 * (S + S.transpose());

        return innovation.transpose() * S.ldlt().solve(innovation);
    }

    void set_eta(double eta) {
        if (!(eta > 0.0) || !std::isfinite(eta)) {
            throw std::invalid_argument("eta must be positive and finite.");
        }
        eta_ = eta;
    }

    void set_lambda(double lambda) {
        if (!(lambda > 0.0) || !std::isfinite(lambda)) {
            throw std::invalid_argument("lambda must be positive and finite.");
        }
        lambda_ = lambda;
    }

    void set_omega(double omega) {
        if (!(omega > 0.0) || !std::isfinite(omega)) {
            throw std::invalid_argument("omega must be positive and finite.");
        }
        omega_ = omega;
    }

    void set_intensity_forgetting_factor(double delta_gamma) {
        if (!(delta_gamma > 0.0 && delta_gamma <= 1.0) || !std::isfinite(delta_gamma)) {
            throw std::invalid_argument("delta_gamma must be in (0, 1].");
        }
        delta_gamma_ = delta_gamma;
    }

    void set_intensity_growth(double gamma_growth) {
        if (!(gamma_growth > 0.0) || !std::isfinite(gamma_growth)) {
            throw std::invalid_argument("gamma_growth must be positive and finite.");
        }
        gamma_growth_ = gamma_growth;
    }

    void set_extent_forgetting_factor(double delta_extent) {
        if (!(delta_extent > 0.0 && delta_extent <= 1.0) || !std::isfinite(delta_extent)) {
            throw std::invalid_argument("delta_extent must be in (0, 1].");
        }
        delta_extent_ = delta_extent;
    }

    /// Exponent applied to per-cell intensities when computing the
    /// measurement centroid: z_bar = (positions * w^p) / sum(w^p).
    /// p = 1 reproduces the standard intensity-weighted centroid.
    /// p > 1 pulls the centroid harder toward hot spots; p < 1 relaxes
    /// it toward the geometric mean of the detection cells.
    void set_centroid_power(double centroid_power) {
        if (!(centroid_power > 0.0) || !std::isfinite(centroid_power)) {
            throw std::invalid_argument("centroid_power must be positive and finite.");
        }
        centroid_power_ = centroid_power;
    }

private:
    double eta_ = 4.0;
    double lambda_ = 4.0;
    double omega_ = 5.0;
    double delta_gamma_ = 0.9;
    double gamma_growth_ = 1.0;
    double delta_extent_ = 0.9;
    double centroid_power_ = 1.0;

    static void unpack_measurement(
        const Eigen::VectorXd& measurement,
        int d,
        Eigen::MatrixXd& positions,
        Eigen::VectorXd& weights) {

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

        throw std::invalid_argument("IGGIWEKF measurement must be length d or packed length (d+1)*N.");
    }

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
        double out = static_cast<double>(d) * (static_cast<double>(d) - 1.0) * 0.25 * std::log(M_PI);
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
            + log_det
            + quad
        );
    }

    static double log_wishart_extent(
        const Eigen::MatrixXd& Z,
        const Eigen::MatrixXd& X,
        double omega) {

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
};

}

namespace brew::filters {
// Concrete filter used for this model (RFS devirtualization).
template <typename Scalar, int D, int De>
struct default_filter<models::IGGIW<Scalar, D, De>> { using type = IGGIWEKF<Scalar, D, De>; };
}  // namespace brew::filters
