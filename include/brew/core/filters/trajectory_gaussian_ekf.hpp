#pragma once

#include "brew/core/filters/filter.hpp"
#include "brew/core/models/trajectory.hpp"
#include "brew/core/models/gaussian.hpp"
#include <cmath>
#include <memory>
#include <utility>

namespace brew::filters {

/// EKF for Gaussian trajectory distributions.
// @mex filter
// @mex_name TrajectoryGaussianEKF
// @mex_dist TrajectoryGaussian
// @mex_setters window_size:int

template <typename Scalar = double, int D = Eigen::Dynamic, int MaxHistory = Eigen::Dynamic>
class TrajectoryGaussianEKF
    : public Filter<models::Trajectory<models::Gaussian<Scalar, D>, MaxHistory>> {
public:
    using InnerDist = models::Gaussian<Scalar, D>;
    using Dist = models::Trajectory<InnerDist, MaxHistory>;
    using TrajectoryType = Dist;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using typename Base::DynamicsType;

    TrajectoryGaussianEKF() = default;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TrajectoryGaussianEKF<Scalar, D, MaxHistory>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->l_window_ = l_window_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {

        Dist result = prev;
        const int sd = prev.state_dim;

        // Dynamics: predict the new step from prev's current last state
        Eigen::VectorXd prev_last_state = prev.get_last_state();
        Eigen::VectorXd next_state = this->dyn_obj_->propagate_state(dt, prev_last_state);
        Eigen::MatrixXd F = this->dyn_obj_->get_state_mat(dt, prev_last_state);

        // Advance ring buffer (slides if at cap); new tail slot is zeroed
        const int cap_hint = Dist::fixed_history ? MaxHistory : l_window_;
        result.advance_window(cap_hint);

        const int last = result.last_index();
        const int prev_last = last - 1;

        // Stacked covariance: fill new last row/col and auto-cov
        // cov[last, k]    = F * cov[prev_last, k]           for k < last
        // cov[last, last] = F * cov[prev_last, prev_last] * F^T + Q
        if (prev_last >= 0) {
            for (int k = 0; k < last; ++k) {
                result.cov_at(last, k) = F * result.cov_at(prev_last, k);
                result.cov_at(k, last) = result.cov_at(last, k).transpose();
            }
            result.cov_at(last, last) =
                F * result.cov_at(prev_last, prev_last) * F.transpose()
                + this->process_noise_;
        } else {
            // First step after an empty trajectory: only process noise.
            result.cov_at(last, last) = this->process_noise_;
        }

        // Stacked mean + history marginal
        result.mean_at(last) = next_state;
        result.history_at(last) = InnerDist(
            next_state,
            Eigen::MatrixXd(result.cov_at(last, last))
        );

        return result;
    }

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const Dist& predicted) const override {

        const int sd = predicted.state_dim;
        const int last = predicted.last_index();
        const int live = predicted.stacked_size();

        const Eigen::VectorXd prev_state = predicted.get_last_state();

        // Measurement model evaluated at the last state
        Eigen::VectorXd est_meas = this->estimate_measurement(prev_state);
        Eigen::MatrixXd H = this->get_measurement_matrix(prev_state);
        const int m = static_cast<int>(H.rows());

        // H_dot selects the last state block from the live stacked state.
        Eigen::MatrixXd H_dot = Eigen::MatrixXd::Zero(m, live);
        H_dot.block(0, live - sd, m, sd) = H;

        // Kalman update on the live slice
        Eigen::MatrixXd P_live = predicted.covariance().topLeftCorner(live, live);
        Eigen::VectorXd mean_live = predicted.mean().head(live);

        Eigen::VectorXd epsilon = measurement - est_meas;
        Eigen::MatrixXd S = H_dot * P_live * H_dot.transpose() + this->measurement_noise_;
        Eigen::MatrixXd K = P_live * H_dot.transpose() * S.inverse();

        Eigen::VectorXd new_mean_live = mean_live + K * epsilon;
        Eigen::MatrixXd new_P_live = P_live - K * H_dot * P_live;

        // Likelihood
        double log_det_S = S.ldlt().vectorD().array().log().sum();
        double mahal = epsilon.transpose() * S.ldlt().solve(epsilon);
        double log_likelihood =
            -0.5 * (static_cast<int>(measurement.size()) * std::log(2.0 * M_PI)
                    + log_det_S + mahal);
        double likelihood = std::exp(log_likelihood);

        // Write corrected slice back into stacked storage; refresh history
        Dist result = predicted;
        result.mean().head(live) = new_mean_live;
        result.covariance().topLeftCorner(live, live) = new_P_live;

        for (int i = 0; i < predicted.window_size(); ++i) {
            result.history_at(i).mean() = result.mean_at(i);
        }
        result.history_at(last).covariance() =
            Eigen::MatrixXd(result.cov_at(last, last));

        return { std::move(result), likelihood };
    }

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const Dist& predicted) const override {

        const Eigen::VectorXd state = predicted.get_last_state();
        const Eigen::MatrixXd P = predicted.get_last_cov();

        const Eigen::VectorXd z_hat = this->estimate_measurement(state);
        const Eigen::MatrixXd H = this->get_measurement_matrix(state);

        Eigen::VectorXd nu = measurement - z_hat;
        Eigen::MatrixXd S = H * P * H.transpose() + this->measurement_noise_;
        S = 0.5 * (S + S.transpose());

        return nu.transpose() * S.ldlt().solve(nu);
    }

    void set_window_size(int L) { l_window_ = L; }
    [[nodiscard]] int window_size() const { return l_window_; }

private:
    int l_window_ = 50;  // only used when MaxHistory is Dynamic
};

} // namespace brew::filters
