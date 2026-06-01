#pragma once
#include "brew/shared/filter_traits.hpp"

#include "brew/shared/filter_base.hpp"
#include "brew/gaussian/gaussian_model.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <utility>

namespace brew::filters {

/// Unscented Kalman Filter for Gaussian distributions.
///
/// Propagates a deterministic set of sigma points through the (possibly
/// nonlinear) dynamics propagate_state() and measurement function h() via the
/// unscented transform, instead of linearizing with Jacobians like the EKF.
/// For genuinely nonlinear models (constant-turn, body-frame, 3D-Euler) this is
/// typically more accurate than the EKF; for linear models it reduces to the
/// Kalman filter and matches the EKF up to numerical precision.
///
/// Scaling parameters: alpha (spread of sigma points, small e.g. 1e-3), beta
/// (prior knowledge of the distribution; 2 is optimal for Gaussians), kappa
/// (secondary scaling, usually 0).
///
/// NOTE: not yet registered as an @mex filter. The RFS hot path is devirtualized
/// around default_filter<Gaussian> == EKF (set_filter downcasts to EKF), so a UKF
/// cannot be dropped into a PHD/GLMB until the RFS is generalized over the filter
/// type. It is fully usable as a standalone single-object filter today.
template <typename Scalar = double, int D = Eigen::Dynamic>
class UKF : public Filter<models::Gaussian<Scalar, D>> {
public:
    using Dist = models::Gaussian<Scalar, D>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    // Local aliases so the static factories (::Zero) parse on dependent types.
    using StateVector      = typename Base::StateVector;
    using StateMatrix      = typename Base::StateMatrix;
    using MeasVector       = typename Base::MeasVector;
    using MeasNoiseMatrix  = typename Base::MeasNoiseMatrix;
    using KalmanGainMatrix = typename Base::KalmanGainMatrix;

    UKF() = default;

    void set_alpha(Scalar a) { alpha_ = a; }
    void set_beta(Scalar b)  { beta_ = b; }
    void set_kappa(Scalar k) { kappa_ = k; }

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<UKF<Scalar, D>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->alpha_ = alpha_;
        c->beta_ = beta_;
        c->kappa_ = kappa_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        const int n = static_cast<int>(prev.mean().size());
        const SigmaSet s = sigma_points(prev.mean(), prev.covariance());

        // Propagate each sigma point through the (nonlinear) dynamics.
        DynMatrix prop(n, 2 * n + 1);
        for (int i = 0; i < 2 * n + 1; ++i) {
            typename Base::StateVector xi = s.pts.col(i);
            prop.col(i) = this->dyn_obj_->propagate_state(dt, xi);
        }

        StateVector mean = StateVector::Zero(n);
        for (int i = 0; i < 2 * n + 1; ++i) mean += s.Wm(i) * prop.col(i);

        typename Base::StateMatrix cov = this->process_noise_;
        for (int i = 0; i < 2 * n + 1; ++i) {
            DynVector d = prop.col(i) - mean;
            cov += s.Wc(i) * (d * d.transpose());
        }

        return Dist(std::move(mean), std::move(cov));
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const int n = static_cast<int>(predicted.mean().size());
        const SigmaSet s = sigma_points(predicted.mean(), predicted.covariance());

        DynMatrix Z;
        typename Base::MeasVector z_hat;
        typename Base::MeasNoiseMatrix S;
        DynMatrix Pxz;
        measurement_transform(s, predicted.mean(), Z, z_hat, S, Pxz);

        typename Base::MeasVector innovation = measurement - z_hat;
        typename Base::KalmanGainMatrix K = Pxz * S.inverse();

        typename Base::StateVector updated_mean = predicted.mean() + K * innovation;
        typename Base::StateMatrix updated_cov = predicted.covariance() - K * S * K.transpose();

        const int m = static_cast<int>(measurement.size());
        double log_det = S.ldlt().vectorD().array().log().sum();
        double mahal = innovation.transpose() * S.ldlt().solve(innovation);
        double log_likelihood = -0.5 * (m * std::log(2.0 * M_PI) + log_det + mahal);

        return {
            Dist(std::move(updated_mean), std::move(updated_cov)),
            std::exp(log_likelihood)
        };
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        const SigmaSet s = sigma_points(predicted.mean(), predicted.covariance());
        DynMatrix Z;
        typename Base::MeasVector z_hat;
        typename Base::MeasNoiseMatrix S;
        DynMatrix Pxz;
        measurement_transform(s, predicted.mean(), Z, z_hat, S, Pxz);

        typename Base::MeasVector innovation = measurement - z_hat;
        return innovation.transpose() * S.ldlt().solve(innovation);
    }

private:
    using DynVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using DynMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    struct SigmaSet {
        DynMatrix pts;   // n x (2n+1) sigma points (columns)
        DynVector Wm;    // mean weights
        DynVector Wc;    // covariance weights
    };

    Scalar alpha_ = Scalar(1e-3);
    Scalar beta_  = Scalar(2);
    Scalar kappa_ = Scalar(0);

    /// Build the 2n+1 sigma points and unscented weights for (mean, cov).
    [[nodiscard]] SigmaSet sigma_points(
        const typename Base::StateVector& mean,
        const typename Base::StateMatrix& cov) const {

        const int n = static_cast<int>(mean.size());
        const Scalar lambda = alpha_ * alpha_ * (n + kappa_) - n;
        const Scalar c = n + lambda;

        SigmaSet s;
        s.Wm.resize(2 * n + 1);
        s.Wc.resize(2 * n + 1);
        s.Wm(0) = lambda / c;
        s.Wc(0) = lambda / c + (Scalar(1) - alpha_ * alpha_ + beta_);
        for (int i = 1; i <= 2 * n; ++i) s.Wm(i) = s.Wc(i) = Scalar(1) / (2 * c);

        // Matrix square root of c*cov via Cholesky (columns are the +/- offsets).
        Eigen::LLT<DynMatrix> llt(c * cov);
        DynMatrix L = llt.matrixL();

        s.pts.resize(n, 2 * n + 1);
        s.pts.col(0) = mean;
        for (int i = 0; i < n; ++i) {
            s.pts.col(1 + i)     = mean + L.col(i);
            s.pts.col(1 + n + i) = mean - L.col(i);
        }
        return s;
    }

    /// Push sigma points through the measurement function and form the
    /// predicted measurement, innovation covariance S, and cross-covariance Pxz.
    void measurement_transform(
        const SigmaSet& s,
        const typename Base::StateVector& state_mean,
        DynMatrix& Z,
        typename Base::MeasVector& z_hat,
        typename Base::MeasNoiseMatrix& S,
        DynMatrix& Pxz) const {

        const int npts = static_cast<int>(s.pts.cols());
        const int n = static_cast<int>(s.pts.rows());
        typename Base::StateVector x0 = s.pts.col(0);
        const int m = static_cast<int>(this->estimate_measurement(x0).size());

        Z.resize(m, npts);
        for (int i = 0; i < npts; ++i) {
            typename Base::StateVector xi = s.pts.col(i);
            Z.col(i) = this->estimate_measurement(xi);
        }

        z_hat = MeasVector::Zero(m);
        for (int i = 0; i < npts; ++i) z_hat += s.Wm(i) * Z.col(i);

        S = this->measurement_noise_;
        Pxz = DynMatrix::Zero(n, m);
        for (int i = 0; i < npts; ++i) {
            DynVector dz = Z.col(i) - z_hat;
            DynVector dx = s.pts.col(i) - state_mean;
            S += s.Wc(i) * (dz * dz.transpose());
            Pxz += s.Wc(i) * (dx * dz.transpose());
        }
    }
};

} // namespace brew::filters
