#include <gtest/gtest.h>
#include "brew/gaussian/filters/ukf.hpp"
#include "brew/gaussian/filters/ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"
#include <algorithm>
#include <cmath>

using namespace brew::filters;
using namespace brew::models;
using namespace brew::dynamics;

TEST(UKFTest, PredictStep) {
    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    UKF<> ukf;
    ukf.set_dynamics(dyn);

    Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
    Eigen::MatrixXd Q_in = 0.01 * Eigen::MatrixXd::Identity(G.cols(), G.cols());
    ukf.set_process_noise(G * Q_in * G.transpose());

    Eigen::VectorXd mean(4);
    mean << 0, 0, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    Gaussian prior(mean, cov);
    auto predicted = ukf.predict(1.0, prior);

    EXPECT_NEAR(predicted.mean()(0), 1.0, 1e-9);
    EXPECT_NEAR(predicted.mean()(1), 1.0, 1e-9);
    EXPECT_GT(predicted.covariance()(0, 0), cov(0, 0));
}

TEST(UKFTest, CorrectStep) {
    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    UKF<> ukf;
    ukf.set_dynamics(dyn);

    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    ukf.set_measurement_jacobian(H);
    ukf.set_measurement_noise(Eigen::MatrixXd::Identity(2, 2) * 0.1);

    Eigen::VectorXd mean(4);
    mean << 0, 0, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 10.0;
    Gaussian predicted(mean, cov);

    Eigen::VectorXd z(2);
    z << 0.5, 0.5;

    auto [corrected, likelihood] = ukf.correct(z, predicted);

    EXPECT_GT(corrected.mean()(0), 0.0);
    EXPECT_LT(corrected.mean()(0), 0.5);
    EXPECT_GT(likelihood, 0.0);
}

TEST(UKFTest, MatchesEKFOnLinearModel) {
    // For linear dynamics + linear measurement the unscented transform is exact,
    // so the UKF must reduce to the Kalman filter and agree with the EKF to
    // numerical precision on both predict and correct.
    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
    Eigen::MatrixXd Q = G * (0.05 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose();
    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    Eigen::MatrixXd R = 0.2 * Eigen::MatrixXd::Identity(2, 2);

    EKF<> ekf;
    ekf.set_dynamics(dyn);
    ekf.set_process_noise(Q);
    ekf.set_measurement_jacobian(H);
    ekf.set_measurement_noise(R);

    UKF<> ukf;
    ukf.set_dynamics(dyn);
    ukf.set_process_noise(Q);
    ukf.set_measurement_jacobian(H);
    ukf.set_measurement_noise(R);

    Eigen::VectorXd mean(4);
    mean << 1, -2, 0.5, 0.3;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 2.0;
    cov(0, 2) = cov(2, 0) = 0.3;
    Gaussian prior(mean, cov);

    auto pe = ekf.predict(1.0, prior);
    auto pu = ukf.predict(1.0, prior);
    EXPECT_LT((pe.mean() - pu.mean()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((pe.covariance() - pu.covariance()).cwiseAbs().maxCoeff(), 1e-6);

    Eigen::VectorXd z(2);
    z << 1.4, -1.7;
    auto [ce, le] = ekf.correct(z, pe);
    auto [cu, lu] = ukf.correct(z, pu);
    EXPECT_LT((ce.mean() - cu.mean()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((ce.covariance() - cu.covariance()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_NEAR(le, lu, 1e-6 * std::max(1.0, le));
}
