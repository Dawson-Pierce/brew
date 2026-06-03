#include <gtest/gtest.h>
#include "brew/ggiw/filters/ggiw_ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"

using namespace brew;

class GGIWEKFTest : public ::testing::Test {
protected:
    void SetUp() override {

        auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
        filter.set_dynamics(dyn);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        filter.set_measurement_jacobian(H);

        filter.set_process_noise(0.1 * Eigen::MatrixXd::Identity(4, 4));
        filter.set_measurement_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
    }

    filters::GGIWEKF<> filter;
};

TEST_F(GGIWEKFTest, PredictDecay) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIW<> ggiw(10.0, 5.0, mean, cov, 10.0, V);
    filter.set_temporal_decay(1.0);
    filter.set_forgetting_factor(1.0);

    auto pred = filter.predict(1.0, ggiw);

    EXPECT_NEAR(pred.mean()(0), 1.0, 1e-10);
    EXPECT_NEAR(pred.mean()(1), 0.0, 1e-10);

    EXPECT_GT(pred.covariance()(0, 0), cov(0, 0));

    EXPECT_DOUBLE_EQ(pred.alpha(), 10.0);
    EXPECT_DOUBLE_EQ(pred.beta(), 5.0);
}

TEST_F(GGIWEKFTest, CorrectAndLikelihood) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 50.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIW<> ggiw(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::VectorXd meas(2);
    meas << 0.1, 0.1;

    auto [corrected, likelihood] = filter.correct(meas, ggiw);

    EXPECT_GT(corrected.mean()(0), 0.0);
    EXPECT_GT(corrected.mean()(1), 0.0);

    EXPECT_LT(corrected.covariance()(0, 0), cov(0, 0));

    EXPECT_GT(likelihood, 0.0);

    EXPECT_DOUBLE_EQ(corrected.alpha(), 11.0);
    EXPECT_DOUBLE_EQ(corrected.beta(), 6.0);

    EXPECT_DOUBLE_EQ(corrected.v(), 11.0);
}

TEST_F(GGIWEKFTest, Gate) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIW<> ggiw(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::VectorXd meas_close(2);
    meas_close << 0.1, 0.1;
    double gate_close = filter.gate(meas_close, ggiw);
    EXPECT_LT(gate_close, 1.0);

    Eigen::VectorXd meas_far(2);
    meas_far << 100.0, 100.0;
    double gate_far = filter.gate(meas_far, ggiw);
    EXPECT_GT(gate_far, 100.0);
}
