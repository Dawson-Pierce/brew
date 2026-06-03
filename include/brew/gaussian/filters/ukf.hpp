#pragma once

#include "brew/shared/filter_traits.hpp"

#include "brew/shared/filter_base.hpp"
#include "brew/gaussian/gaussian_model.hpp"
#include "brew/gaussian/filters/gaussian_lti_predict.hpp"
#include "brew/assert.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <utility>

namespace brew::filters {

// @mex filter
// @mex_name UKF
// @mex_dist Gaussian
// @mex_setters alpha:scalar, beta:scalar, kappa:scalar
template <typename Scalar = double, int D = Eigen::Dynamic>
class UKF : public Filter<models::Gaussian<Scalar, D>> {
public:
    using Dist = models::Gaussian<Scalar, D>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

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

        if (this->dyn_obj_ && this->dyn_obj_->is_lti()) {
            const StateMatrix F = this->dyn_obj_->get_state_mat(dt, prev.mean());
            return Dist(F * prev.mean(),
                        F * prev.covariance() * F.transpose() + this->process_noise_);
        }

        const int n = static_cast<int>(prev.mean().size());
        const SigmaSet s = sigma_points(prev.mean(), prev.covariance());

        DynMatrix prop(n, 2 * n + 1);
        for (int i = 0; i < 2 * n + 1; ++i) {
            typename Base::StateVector xi = s.pts.col(i);
            prop.col(i) = this->dyn_obj_->propagate_state(dt, xi);
        }

        StateVector mean = prop.col(0);
        for (int i = 1; i < 2 * n + 1; ++i) mean += s.Wm(i) * (prop.col(i) - prop.col(0));

        typename Base::StateMatrix cov = this->process_noise_;
        for (int i = 0; i < 2 * n + 1; ++i) {
            DynVector d = prop.col(i) - mean;
            cov += s.Wc(i) * (d * d.transpose());
        }

        return Dist(std::move(mean), std::move(cov));
    }

    void predict_batch_dynamic(
        double dt, models::Mixture<Dist, Eigen::Dynamic>& mix) const override {
        if (mix.size() == 0) return;
        if (this->dyn_obj_ && this->dyn_obj_->is_lti()) {
            detail::gaussian_lti_batch_predict(dt, *this->dyn_obj_, this->process_noise_, mix);
        } else {
            const std::size_t K = mix.size();
            for (std::size_t k = 0; k < K; ++k)
                mix.component(k) = this->predict(dt, mix.component(k));
        }
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
        DynMatrix pts;
        DynVector Wm;
        DynVector Wc;
    };

    Scalar alpha_ = Scalar(1);
    Scalar beta_  = Scalar(2);
    Scalar kappa_ = Scalar(0);

    [[nodiscard]] SigmaSet sigma_points(
        const typename Base::StateVector& mean,
        const typename Base::StateMatrix& cov) const {

        const int n = static_cast<int>(mean.size());
        const Scalar lambda = alpha_ * alpha_ * (n + kappa_) - n;
        const Scalar c = n + lambda;
        BREW_ASSERT(c > Scalar(0),
            "UKF: n + lambda must be positive; check alpha/kappa (need alpha^2*(n+kappa) > 0)");

        SigmaSet s;
        s.Wm.resize(2 * n + 1);
        s.Wc.resize(2 * n + 1);
        s.Wm(0) = lambda / c;
        s.Wc(0) = lambda / c + (Scalar(1) - alpha_ * alpha_ + beta_);
        for (int i = 1; i <= 2 * n; ++i) s.Wm(i) = s.Wc(i) = Scalar(1) / (2 * c);

        DynMatrix P = c * cov;
        P = Scalar(0.5) * (P + P.transpose());
        Eigen::LLT<DynMatrix> llt(P);
        if (llt.info() != Eigen::Success) {
            const Scalar jitter = Scalar(1e-9) * (P.diagonal().cwiseAbs().maxCoeff() + Scalar(1));
            P.diagonal().array() += jitter;
            llt.compute(P);
        }
        DynMatrix L = llt.matrixL();

        s.pts.resize(n, 2 * n + 1);
        s.pts.col(0) = mean;
        for (int i = 0; i < n; ++i) {
            s.pts.col(1 + i)     = mean + L.col(i);
            s.pts.col(1 + n + i) = mean - L.col(i);
        }
        return s;
    }

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

        z_hat = Z.col(0);
        for (int i = 1; i < npts; ++i) z_hat += s.Wm(i) * (Z.col(i) - Z.col(0));

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

}
