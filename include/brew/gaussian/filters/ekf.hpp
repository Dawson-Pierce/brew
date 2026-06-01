#pragma once
#include "brew/shared/filter_traits.hpp"

#include "brew/shared/filter_base.hpp"
#include "brew/gaussian/gaussian_model.hpp"
#include "brew/assert.hpp"
#include <cmath>
#include <memory>
#include <utility>

namespace brew::filters {

/// Extended Kalman Filter for Gaussian distributions.
// @mex filter
// @mex_name EKF
// @mex_dist Gaussian

template <typename Scalar = double, int D = Eigen::Dynamic>
class EKF : public Filter<models::Gaussian<Scalar, D>> {
public:
    using Dist = models::Gaussian<Scalar, D>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    EKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<EKF<Scalar, D>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        typename Base::StateMatrix F = this->dyn_obj_->get_state_mat(dt, prev.mean());
        // const auto G = this->dyn_obj_->get_input_mat(dt, prev.mean());

        typename Base::StateVector predicted_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
        typename Base::StateMatrix predicted_cov = F * prev.covariance() * F.transpose()
                                                   + this->process_noise_;

        return Dist(std::move(predicted_mean), std::move(predicted_cov));
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        typename Base::MeasMatrix H = this->get_measurement_matrix(predicted.mean());
        typename Base::MeasVector z_hat = this->estimate_measurement(predicted.mean());

        typename Base::MeasVector innovation = measurement - z_hat;
        typename Base::MeasNoiseMatrix S = H * predicted.covariance() * H.transpose()
                                           + this->measurement_noise_;

        typename Base::KalmanGainMatrix K = predicted.covariance() * H.transpose() * S.inverse();

        typename Base::StateVector updated_mean = predicted.mean() + K * innovation;
        const auto state_dim = predicted.mean().size();
        typename Base::StateMatrix I = Base::StateMatrix::Identity(state_dim, state_dim);
        typename Base::StateMatrix updated_cov = (I - K * H) * predicted.covariance();

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

        typename Base::MeasMatrix H = this->get_measurement_matrix(predicted.mean());
        typename Base::MeasVector z_hat = this->estimate_measurement(predicted.mean());

        typename Base::MeasVector innovation = measurement - z_hat;
        typename Base::MeasNoiseMatrix S = H * predicted.covariance() * H.transpose()
                                           + this->measurement_noise_;

        return innovation.transpose() * S.ldlt().solve(innovation);
    }

    /// LTI fast path: for linear time-invariant dynamics the state transition F
    /// depends only on dt, so build it ONCE and apply that shared F to every
    /// component, instead of rebuilding F per target (predict() builds F twice
    /// per component: once explicitly and once inside propagate_state). The math
    /// is identical to predict() (same F, same F*x and F*P*F^T + Q), so results
    /// are bit-for-bit unchanged. Falls back to the per-component loop otherwise.
    template <int N>
    void predict_batch(double dt, models::Mixture<Dist, N>& mix) const {
        const std::size_t K = mix.size();
        if (K == 0) return;
        if (!this->dyn_obj_ || !this->dyn_obj_->is_lti()) {
            for (std::size_t k = 0; k < K; ++k)
                mix.component(k) = this->predict(dt, mix.component(k));
            return;
        }
        const typename Base::StateMatrix F  = this->dyn_obj_->get_state_mat(dt, mix.component(0).mean());
        // Guard the is_lti() contract (debug-only; compiles out under NDEBUG):
        // a state-independent F must be identical regardless of which component's
        // mean it is built from, so building it once from component(0) is exact.
        // Catches a dynamics that wrongly reports is_lti() while reading the state.
        BREW_ASSERT(K < 2 ||
            (this->dyn_obj_->get_state_mat(dt, mix.component(K - 1).mean()).array() == F.array()).all(),
            "EKF::predict_batch: dynamics reports is_lti() but get_state_mat depends on state");
        const typename Base::StateMatrix Ft = F.transpose();
        for (std::size_t k = 0; k < K; ++k) {
            auto& c = mix.component(k);
            typename Base::StateVector pm = F * c.mean();
            typename Base::StateMatrix pc = F * c.covariance() * Ft + this->process_noise_;
            c.mean() = std::move(pm);
            c.covariance() = std::move(pc);
        }
    }
};

} // namespace brew::filters

namespace brew::filters {
// Concrete filter used for this model (RFS devirtualization).
template <typename Scalar, int D>
struct default_filter<models::Gaussian<Scalar, D>> { using type = EKF<Scalar, D>; };
}  // namespace brew::filters
