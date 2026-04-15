#pragma once

#include "brew/core/filters/filter.hpp"
#include "brew/core/models/gaussian.hpp"
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

        const auto F = this->dyn_obj_->get_state_mat(dt, prev.mean());
        const auto G = this->dyn_obj_->get_input_mat(dt, prev.mean());

        Eigen::VectorXd predicted_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
        Eigen::MatrixXd predicted_cov = F * prev.covariance() * F.transpose()
                                        + G * this->process_noise_ * G.transpose();

        return Dist(std::move(predicted_mean), std::move(predicted_cov));
    }

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const Dist& predicted) const override {

        const auto H = this->get_measurement_matrix(predicted.mean());
        const auto z_hat = this->estimate_measurement(predicted.mean());

        Eigen::VectorXd innovation = measurement - z_hat;
        Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + this->measurement_noise_;

        Eigen::MatrixXd K = predicted.covariance() * H.transpose() * S.inverse();

        Eigen::VectorXd updated_mean = predicted.mean() + K * innovation;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(predicted.mean().size(), predicted.mean().size());
        Eigen::MatrixXd updated_cov = (I - K * H) * predicted.covariance();

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
        const Eigen::VectorXd& measurement,
        const Dist& predicted) const override {

        const auto H = this->get_measurement_matrix(predicted.mean());
        const auto z_hat = this->estimate_measurement(predicted.mean());

        Eigen::VectorXd innovation = measurement - z_hat;
        Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + this->measurement_noise_;

        return innovation.transpose() * S.ldlt().solve(innovation);
    }
};

} // namespace brew::filters
