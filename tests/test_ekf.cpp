#include <gtest/gtest.h>
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew::filters;
using namespace brew::distributions;
using namespace brew::dynamics;

TEST(EKF, PredictStep) {
    auto dyn = std::make_shared<Integrator2D>();
    EKF ekf;
    ekf.set_dynamics(dyn);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
    ekf.set_process_noise(Q);

    Eigen::VectorXd mean(4);
    mean << 0, 0, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    Gaussian prior(mean, cov);
    auto predicted = ekf.predict(1.0, prior);

    // Mean should propagate: x=1, y=1
    EXPECT_NEAR(predicted.mean()(0), 1.0, 1e-10);
    EXPECT_NEAR(predicted.mean()(1), 1.0, 1e-10);

    // Covariance should grow
    EXPECT_GT(predicted.covariance()(0, 0), cov(0, 0));
}

TEST(EKF, CorrectStep) {
    auto dyn = std::make_shared<Integrator2D>();
    EKF ekf;
    ekf.set_dynamics(dyn);

    // Position-only measurement: H = [I2 02]
    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    ekf.set_measurement_jacobian(H);
    ekf.set_measurement_noise(Eigen::MatrixXd::Identity(2, 2) * 0.1);

    Eigen::VectorXd mean(4);
    mean << 0, 0, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 10.0;
    Gaussian predicted(mean, cov);

    Eigen::VectorXd z(2);
    z << 0.5, 0.5;

    auto [corrected, likelihood] = ekf.correct(z, predicted);

    // Corrected mean should move toward measurement
    EXPECT_GT(corrected.mean()(0), 0.0);
    EXPECT_LT(corrected.mean()(0), 0.5);

    // Likelihood should be positive
    EXPECT_GT(likelihood, 0.0);
}
