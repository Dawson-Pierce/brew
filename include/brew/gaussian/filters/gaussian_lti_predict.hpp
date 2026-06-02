#pragma once

#include "brew/dynamics/dynamics_base.hpp"
#include "brew/gaussian/gaussian_model.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/assert.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <utility>

namespace brew::filters::detail {

/// Shared LTI batch predict for Gaussian filters.
///
/// Under linear time-invariant dynamics the unscented transform is EXACT and
/// equals the EKF linearization, so the EKF and UKF predict steps are identical
/// and both reduce to the Kalman predict. Build the state transition F once and
/// apply mean' = F*mean, cov' = F*P*F^T + Q to every component — no Jacobian
/// rebuild per target, no sigma points. Both EKF and UKF call this when their
/// dynamics report is_lti(); only the nonlinear path differs between them.
///
/// Precondition: the caller has checked dyn.is_lti(); the mix may be empty.
template <typename Scalar, int D, int N>
inline void gaussian_lti_batch_predict(
    double dt,
    const dynamics::DynamicsBase<Scalar, D>& dyn,
    const Eigen::Matrix<Scalar, D, D>& Q,
    models::Mixture<models::Gaussian<Scalar, D>, N>& mix) {

    const std::size_t K = mix.size();
    if (K == 0) return;

    using Vec = Eigen::Matrix<Scalar, D, 1>;
    using Mat = Eigen::Matrix<Scalar, D, D>;

    const Mat F = dyn.get_state_mat(dt, mix.component(0).mean());
    // Debug guard (compiled out under NDEBUG): a genuinely state-independent F is
    // the same regardless of which component's mean it is built from, so building
    // it once from component(0) is exact. Catches a dynamics that wrongly reports
    // is_lti() while actually reading the state.
    BREW_ASSERT(K < 2 ||
        (dyn.get_state_mat(dt, mix.component(K - 1).mean()).array() == F.array()).all(),
        "gaussian_lti_batch_predict: dynamics reports is_lti() but get_state_mat depends on state");
    const Mat Ft = F.transpose();

    for (std::size_t k = 0; k < K; ++k) {
        auto& c = mix.component(k);
        Vec pm = F * c.mean();
        Mat pc = F * c.covariance() * Ft + Q;
        c.mean() = std::move(pm);
        c.covariance() = std::move(pc);
    }
}

} // namespace brew::filters::detail
