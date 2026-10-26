#include <gtest/gtest.h>
#include "brew/ggiw/filters/ggiw_ekf.hpp"
#include "brew/ggiw/merge.hpp"
#include "brew/ggiw/multi_target/phd.hpp"
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

TEST_F(GGIWEKFTest, NonzeroMeasurementNoiseTransformsExtentStatistics) {
    Eigen::Vector4d mean = Eigen::Vector4d::Zero();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Identity();
    Eigen::Matrix2d V = 8.0 * Eigen::Matrix2d::Identity();
    models::GGIW<> ggiw(10.0, 5.0, mean, cov, 10.0, V);
    filter.set_measurement_noise(2.0 * Eigen::Matrix2d::Identity());

    Eigen::VectorXd packed(4);
    packed << 0.0, 0.0, 2.0, 0.0;
    auto [corrected, likelihood] = filter.correct(packed, ggiw);

    // Xhat = 2I, Rhat = 4I and S = 3I. Both the centroid innovation and
    // within-cell scatter must be transformed when measurement noise is nonzero.
    EXPECT_NEAR(corrected.V()(0, 0), 8.0 + 2.0 / 3.0 + 1.0, 1e-10);
    EXPECT_NEAR(corrected.V()(1, 1), 8.0, 1e-10);
    EXPECT_TRUE(std::isfinite(likelihood));
    EXPECT_GT(likelihood, 0.0);
}

TEST_F(GGIWEKFTest, ComponentCorrectionUsesCovarianceIntersection) {
    Eigen::Vector4d predicted_mean = Eigen::Vector4d::Zero();
    Eigen::Vector4d report_mean;
    report_mean << 2.0, 0.0, 4.0, 0.0;
    models::GGIW<> predicted(
        4.0, 2.0, predicted_mean, 2.0 * Eigen::Matrix4d::Identity(),
        10.0, 4.0 * Eigen::Matrix2d::Identity());
    models::GGIW<> report(
        8.0, 4.0, report_mean, 4.0 * Eigen::Matrix4d::Identity(),
        14.0, 16.0 * Eigen::Matrix2d::Identity());
    filter.set_ci_weight(0.5);

    auto [corrected, likelihood] = filter.correct_component(report, predicted);

    EXPECT_TRUE(corrected.covariance().isApprox(
        (8.0 / 3.0) * Eigen::Matrix4d::Identity(), 1e-12));
    EXPECT_TRUE(corrected.mean().isApprox(report_mean / 3.0, 1e-12));
    EXPECT_NEAR(corrected.alpha(), 6.0, 1e-12);
    EXPECT_NEAR(corrected.beta(), 3.0, 1e-12);
    EXPECT_NEAR(corrected.v(), 12.0, 1e-12);
    EXPECT_TRUE(corrected.V().isApprox(10.0 * Eigen::Matrix2d::Identity(), 1e-12));
    EXPECT_TRUE(std::isfinite(likelihood));
    EXPECT_GT(likelihood, 0.0);
}

TEST(GGIWMerge, PreservesMixtureMoments) {
    models::Mixture<models::GGIW<>> mix;
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d V1 = 4.0 * Eigen::Matrix2d::Identity();
    Eigen::Matrix2d V2 = 16.0 * Eigen::Matrix2d::Identity();
    mix.add_component(std::make_unique<models::GGIW<>>(
        4.0, 2.0, Eigen::Vector2d(0.0, 0.0), P, 10.0, V1), 0.25);
    mix.add_component(std::make_unique<models::GGIW<>>(
        18.0, 3.0, Eigen::Vector2d(2.0, 0.0), P, 14.0, V2), 0.75);

    ggiw::merge(mix, 25.0);

    ASSERT_EQ(mix.size(), 1u);
    EXPECT_TRUE(mix.component(0).mean().isApprox(Eigen::Vector2d(1.5, 0.0), 1e-12));
    Eigen::Matrix2d expected_cov = Eigen::Matrix2d::Identity();
    expected_cov(0, 0) += 0.75;
    EXPECT_TRUE(mix.component(0).covariance().isApprox(expected_cov, 1e-12));
    EXPECT_NEAR(mix.component(0).alpha() / mix.component(0).beta(), 5.0, 1e-12);
    const Eigen::Matrix2d expected_extent = mix.component(0).V()
        / (mix.component(0).v() - 6.0);
    EXPECT_TRUE(expected_extent.isApprox(1.75 * Eigen::Matrix2d::Identity(), 1e-12));
}

TEST(GGIWPHD, RetainsDetectedTargetWithZeroGeneratedMeasurements) {
    filters::GGIWEKF<> filter;
    ggiw::PHD<> phd;
    phd.set_filter(filter.clone());
    models::Mixture<models::GGIW<>> intensity;
    intensity.add_component(std::make_unique<models::GGIW<>>(
        10.0, 2.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity(),
        10.0, 4.0 * Eigen::Matrix2d::Identity()), 0.8);
    phd.set_intensity(intensity.clone());
    phd.set_prob_detection(1.0);

    phd.correct(Eigen::MatrixXd(2, 0));

    ASSERT_EQ(phd.intensity().size(), 1u);
    EXPECT_NEAR(phd.intensity().component(0).beta(), 3.0, 1e-12);
    EXPECT_NEAR(phd.intensity().weight(0),
                0.8 * std::pow(2.0 / 3.0, 10.0), 1e-12);
}
