#pragma once

#include "brew/shared/filter_traits.hpp"
#include "brew/shared/parameter_validation.hpp"

#include "brew/shared/filter_base.hpp"
#include "brew/ggiw/ggiw_model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace brew::filters {

namespace detail {

inline Eigen::MatrixXd sqrtm_spd(const Eigen::MatrixXd& M) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
    Eigen::VectorXd d = es.eigenvalues();
    for (int i = 0; i < d.size(); ++i) {
        d(i) = std::sqrt(std::max(d(i), std::numeric_limits<double>::epsilon()));
    }
    return es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
}

inline double logdet_spd(const Eigen::MatrixXd& M) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
    return es.eigenvalues().array()
        .max(std::numeric_limits<double>::min()).log().sum();
}

inline double safe_exp(double value) {
    return std::exp(std::clamp(value, -700.0, 600.0));
}

}

// @mex filter
// @mex_name GGIWEKF
// @mex_dist GGIW
// @mex_setters temporal_decay:scalar, forgetting_factor:scalar, scaling_parameter:scalar, ci_weight:scalar

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class GGIWEKF : public Filter<models::GGIW<Scalar, D, De>> {
public:
    using Dist = models::GGIW<Scalar, D, De>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    GGIWEKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<GGIWEKF<Scalar, D, De>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->eta_ = eta_;
        c->tau_ = tau_;
        c->rho_ = rho_;
        c->ci_weight_ = ci_weight_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        const auto F = this->dyn_obj_->get_state_mat(dt, prev.mean());
        Eigen::VectorXd next_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
        Eigen::MatrixXd next_cov = F * prev.covariance() * F.transpose()
                                 + this->process_noise_;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        double next_alpha = prev.alpha() / eta_;
        double next_beta = prev.beta() / eta_;

        const int d = prev.extent_dim();
        const double dof_floor = 2.0 * d + 2.0;
        double next_v = dof_floor + std::exp(-dt / tau_) * (prev.v() - dof_floor);

        double scale = (next_v - dof_floor) / std::max(prev.v() - dof_floor, 1e-12);
        Eigen::MatrixXd next_V = scale * this->dyn_obj_->propagate_extent(dt, prev.mean(), prev.V());

        return Dist(
            next_alpha, next_beta,
            std::move(next_mean), std::move(next_cov),
            next_v, std::move(next_V));
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const int d = predicted.extent_dim();

        if (measurement.size() < d || measurement.size() % d != 0) {
            throw std::invalid_argument("GGIWEKF measurement must contain one or more packed d-dimensional points.");
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
        // Guard the extent normalizer (the gate() path already does): if v has decayed to the
        // dof floor, dividing by zero yields an inf/NaN extent that then makes the downstream
        // matrix square-root / eigensolver spin. max(.,1e-12) keeps X_hat finite.
        Eigen::MatrixXd X_hat = predicted.V() / std::max(predicted.v() - dof_floor, 1e-12);

        const auto H = this->get_measurement_matrix(predicted.mean());
        const Eigen::VectorXd est_meas = this->estimate_measurement(predicted.mean());
        const Eigen::VectorXd epsilon = mean_meas - est_meas;

        const Eigen::MatrixXd N = epsilon * epsilon.transpose();

        Eigen::MatrixXd R_hat = rho_ * X_hat + this->measurement_noise_;
        Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + R_hat / static_cast<double>(W);

        const Eigen::MatrixXd Sinv = S.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd K = predicted.covariance() * H.transpose() * Sinv;

        Eigen::MatrixXd sqrt_X = detail::sqrtm_spd(X_hat);
        Eigen::MatrixXd sqrt_R = detail::sqrtm_spd(R_hat);
        Eigen::MatrixXd sqrt_S = detail::sqrtm_spd(S);
        Eigen::MatrixXd sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd sqrt_S_inv = sqrt_S.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd N_hat = sqrt_X * sqrt_S_inv * N * sqrt_S_inv.transpose() * sqrt_X.transpose();
        Eigen::MatrixXd Z_hat = sqrt_X * sqrt_R_inv * scatter
                              * sqrt_R_inv.transpose() * sqrt_X.transpose();

        double next_alpha = predicted.alpha() + W;
        double next_beta = predicted.beta() + 1.0;
        Eigen::VectorXd next_mean = predicted.mean() + K * epsilon;
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(
            predicted.covariance().rows(), predicted.covariance().cols());
        const Eigen::MatrixXd A = I - K * H;
        Eigen::MatrixXd next_cov = A * predicted.covariance() * A.transpose()
                                 + K * (R_hat / static_cast<double>(W)) * K.transpose();
        next_cov = 0.5 * (next_cov + next_cov.transpose());
        {   // Keep the kinematic covariance positive-definite. The non-Joseph update P - K H P can
            // lose PD over many corrections on a long-lived track; a non-PD covariance then breaks
            // the Cholesky in merge() and corrupts the component (observed as a spurious dimension
            // blow-up). Clamp the spectrum to stay SPD.
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(next_cov);
            if (es.info() == Eigen::Success) {
                Eigen::VectorXd ev = es.eigenvalues().cwiseMax(1e-9);
                next_cov.noalias() = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
            }
        }

        double next_v = predicted.v() + W;
        Eigen::MatrixXd next_V = predicted.V() + N_hat + Z_hat;

        const double v0 = predicted.v();
        const double v1 = next_v;
        const double a0 = predicted.alpha();
        const double a1 = next_alpha;
        const double b0 = predicted.beta();
        const double b1 = next_beta;

        const double log_det_V0 = detail::logdet_spd(predicted.V());
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

        return {
            Dist(next_alpha, next_beta,
                 std::move(next_mean), std::move(next_cov),
                 next_v, std::move(next_V)),
            detail::safe_exp(log_likelihood)
        };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const int d = predicted.extent_dim();
        const double dof_floor = 2.0 * d + 2.0;
        Eigen::MatrixXd X_hat = predicted.V() / std::max(predicted.v() - dof_floor, 1e-12);

        const auto H = this->get_measurement_matrix(predicted.mean());
        const Eigen::VectorXd z_hat = this->estimate_measurement(predicted.mean());
        const Eigen::VectorXd nu = measurement - z_hat;

        Eigen::MatrixXd S = H * predicted.covariance() * H.transpose()
                           + rho_ * X_hat + this->measurement_noise_;
        S = 0.5 * (S + S.transpose());

        return nu.transpose() * S.ldlt().solve(nu);
    }

    // Fuse a GGIW report into a predicted track using covariance intersection. The reports can share
    // information, so treating one posterior as fresh conjugate evidence would double count it.
    [[nodiscard]] CorrectionResult correct_component(const Dist& z, const Dist& predicted) const {
        const double omega = std::clamp(ci_weight_, 0.0, 1.0);
        const Eigen::MatrixXd Pg_inv = predicted.covariance().ldlt().solve(
            Eigen::MatrixXd::Identity(predicted.mean().size(), predicted.mean().size()));
        const Eigen::MatrixXd Pz_inv = z.covariance().ldlt().solve(
            Eigen::MatrixXd::Identity(z.mean().size(), z.mean().size()));
        const Eigen::MatrixXd info = omega * Pg_inv + (1.0 - omega) * Pz_inv;
        Eigen::MatrixXd next_cov = info.ldlt().solve(
            Eigen::MatrixXd::Identity(info.rows(), info.cols()));
        next_cov = 0.5 * (next_cov + next_cov.transpose());
        Eigen::VectorXd next_mean = next_cov * (
            omega * Pg_inv * predicted.mean() + (1.0 - omega) * Pz_inv * z.mean());

        const double next_v = omega * predicted.v() + (1.0 - omega) * z.v();
        Eigen::MatrixXd next_V = omega * predicted.V() + (1.0 - omega) * z.V();
        const double next_alpha = omega * predicted.alpha() + (1.0 - omega) * z.alpha();
        const double next_beta = omega * predicted.beta() + (1.0 - omega) * z.beta();

        Eigen::MatrixXd full_S = predicted.covariance() + z.covariance();
        full_S = 0.5 * (full_S + full_S.transpose());
        const Eigen::VectorXd full_epsilon = z.mean() - predicted.mean();
        const double log_det_S = detail::logdet_spd(full_S);
        const double maha = full_epsilon.dot(full_S.ldlt().solve(full_epsilon));
        const double log_like = -0.5 * (maha + log_det_S
            + predicted.mean().size() * std::log(2.0 * M_PI));

        return {
            Dist(next_alpha, next_beta, std::move(next_mean), std::move(next_cov), next_v, std::move(next_V)),
            detail::safe_exp(log_like)
        };
    }

    [[nodiscard]] double gate_component(const Dist& z, const Dist& predicted) const {
        const Eigen::VectorXd nu = z.mean() - predicted.mean();
        Eigen::MatrixXd S = predicted.covariance() + z.covariance();
        S = 0.5 * (S + S.transpose());
        return nu.transpose() * S.ldlt().solve(nu);
    }

    void set_temporal_decay(double eta) { eta_ = brew::detail::positive_finite(eta, "Temporal decay"); }
    void set_forgetting_factor(double tau) { tau_ = brew::detail::positive_finite(tau, "Forgetting time"); }
    void set_scaling_parameter(double rho) { rho_ = brew::detail::positive_finite(rho, "Extent scale"); }
    void set_ci_weight(double omega) { ci_weight_ = brew::detail::unit_interval(omega, "CI weight"); }

private:
    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;
    double ci_weight_ = 0.5;
};

}

namespace brew::filters {

template <typename Scalar, int D, int De>
struct default_filter<models::GGIW<Scalar, D, De>> { using type = GGIWEKF<Scalar, D, De>; };
}
