#include "brew/filters/ekf.hpp"
#include <cmath>

namespace brew::filters {

std::unique_ptr<Filter<distributions::Gaussian>> EKF::clone() const {
    auto c = std::make_unique<EKF>();
    c->dyn_obj_ = dyn_obj_;
    c->h_ = h_;
    c->H_ = H_;
    c->process_noise_ = process_noise_;
    c->measurement_noise_ = measurement_noise_;
    return c;
}

distributions::Gaussian EKF::predict(
    double dt,
    const distributions::Gaussian& prev) const {

    const auto F = dyn_obj_->get_state_mat(dt, prev.mean());
    const auto G = dyn_obj_->get_input_mat(dt, prev.mean());

    Eigen::VectorXd predicted_mean = dyn_obj_->propagate_state(dt, prev.mean());
    Eigen::MatrixXd predicted_cov = F * prev.covariance() * F.transpose()
                                    + G * process_noise_ * G.transpose();

    return distributions::Gaussian(std::move(predicted_mean), std::move(predicted_cov));
}

EKF::CorrectionResult EKF::correct(
    const Eigen::VectorXd& measurement,
    const distributions::Gaussian& predicted) const {

    const auto H = get_measurement_matrix(predicted.mean());
    const auto z_hat = estimate_measurement(predicted.mean());

    Eigen::VectorXd innovation = measurement - z_hat;
    Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + measurement_noise_;

    Eigen::MatrixXd K = predicted.covariance() * H.transpose() * S.inverse();

    Eigen::VectorXd updated_mean = predicted.mean() + K * innovation;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(predicted.mean().size(), predicted.mean().size());
    Eigen::MatrixXd updated_cov = (I - K * H) * predicted.covariance();

    const int m = static_cast<int>(measurement.size());
    double log_det = S.ldlt().vectorD().array().log().sum();
    double mahal = innovation.transpose() * S.ldlt().solve(innovation);
    double log_likelihood = -0.5 * (m * std::log(2.0 * M_PI) + log_det + mahal);

    return {
        distributions::Gaussian(std::move(updated_mean), std::move(updated_cov)),
        std::exp(log_likelihood)
    };
}

double EKF::gate(
    const Eigen::VectorXd& measurement,
    const distributions::Gaussian& predicted) const {

    const auto H = get_measurement_matrix(predicted.mean());
    const auto z_hat = estimate_measurement(predicted.mean());

    Eigen::VectorXd innovation = measurement - z_hat;
    Eigen::MatrixXd S = H * predicted.covariance() * H.transpose() + measurement_noise_;

    return innovation.transpose() * S.ldlt().solve(innovation);
}

} // namespace brew::filters
