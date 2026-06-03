#include <gtest/gtest.h>
#include "brew/trajectory_ggiw_orientation/filters/trajectory_ggiw_orientation_ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"

using namespace brew;

namespace {
constexpr int kTestWindow = 10;
}

TEST(TrajectoryGGIWOrientationModel, ConstructAndAccessors) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 10.0, 2.0,
          2.0, 5.0;

    models::TrajectoryGGIWOrientation<>g(4, models::GGIWOrientation<>(10.0, 5.0, mean, cov, 10.0, V), kTestWindow);

    EXPECT_EQ(g.current().extent_dim(), 2);
    EXPECT_TRUE(g.is_extended());
    EXPECT_DOUBLE_EQ(g.current().alpha(), 10.0);
    EXPECT_DOUBLE_EQ(g.current().beta(), 5.0);
    EXPECT_DOUBLE_EQ(g.current().v(), 10.0);
    EXPECT_EQ(g.state_dim, 4);
    EXPECT_EQ(g.window_size(), 1);

    EXPECT_EQ(g.current().basis().rows(), 2);
    EXPECT_EQ(g.current().basis().cols(), 2);
    EXPECT_TRUE(g.current().has_eigenvalues());

    EXPECT_EQ(g.get_last_state().size(), 4);
    EXPECT_EQ(g.get_last_cov().rows(), 4);
    EXPECT_EQ(g.mean_history().cols(), 1);
}

TEST(TrajectoryGGIWOrientationModel, Clone) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation<>g(4, models::GGIWOrientation<>(10.0, 5.0, mean, cov, 10.0, V), kTestWindow);

    auto typed_clone = g.clone_typed();
    ASSERT_NE(typed_clone, nullptr);
    EXPECT_DOUBLE_EQ(typed_clone->current().alpha(), 10.0);
    EXPECT_EQ(typed_clone->current().basis().rows(), 2);
    EXPECT_EQ(typed_clone->state_dim, 4);
    EXPECT_TRUE(typed_clone->current().has_eigenvalues());
}

class TrajectoryGGIWOrientationEKFTest : public ::testing::Test {
protected:
    static constexpr int kWindow = kTestWindow;

    void SetUp() override {
        auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
        filter.set_dynamics(dyn);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        filter.set_measurement_jacobian(H);

        filter.set_process_noise(0.1 * Eigen::MatrixXd::Identity(4, 4));
        filter.set_measurement_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
        filter.set_temporal_decay(1.0);
        filter.set_forgetting_factor(1.0);
    }

    filters::TrajectoryGGIWOrientationEKF<> filter;
};

TEST_F(TrajectoryGGIWOrientationEKFTest, PredictGrowsTrajectory) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation<>tg(4, models::GGIWOrientation<>(10.0, 5.0, mean, cov, 10.0, V), kTestWindow);

    auto pred = filter.predict(1.0, tg);

    EXPECT_EQ(pred.window_size(), 2);
    EXPECT_EQ(pred.stacked_size(), 8);

    Eigen::VectorXd last = pred.get_last_state();
    EXPECT_NEAR(last(0), 1.0, 1e-10);

    EXPECT_EQ(pred.current().basis().rows(), 2);
}

TEST_F(TrajectoryGGIWOrientationEKFTest, CorrectPopulatesBasis) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 50.0, 10.0,
         10.0, 30.0;

    models::TrajectoryGGIWOrientation<>tg(4, models::GGIWOrientation<>(10.0, 5.0, mean, cov, 10.0, V), kTestWindow);

    auto pred = filter.predict(1.0, tg);

    Eigen::VectorXd meas(2);
    meas << 1.1, 0.1;
    auto [corrected, likelihood] = filter.correct(meas, pred);

    EXPECT_GT(likelihood, 0.0);

    EXPECT_DOUBLE_EQ(corrected.current().alpha(), pred.current().alpha() + 1.0);
    EXPECT_DOUBLE_EQ(corrected.current().beta(), pred.current().beta() + 1.0);

    EXPECT_DOUBLE_EQ(corrected.current().v(), pred.current().v() + 1.0);

    EXPECT_EQ(corrected.current().basis().rows(), 2);
    EXPECT_EQ(corrected.current().basis().cols(), 2);
    EXPECT_TRUE(corrected.current().has_eigenvalues());

    EXPECT_EQ(corrected.window_size(), pred.window_size());
}

TEST_F(TrajectoryGGIWOrientationEKFTest, MultipleStepsBasisAlignment) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 80.0, 5.0,
          5.0, 20.0;

    models::TrajectoryGGIWOrientation<>tg(4, models::GGIWOrientation<>(10.0, 5.0, mean, cov, 10.0, V), kTestWindow);

    auto pred1 = filter.predict(1.0, tg);
    Eigen::VectorXd meas1(2);
    meas1 << 1.05, 0.05;
    auto [corr1, lik1] = filter.correct(meas1, pred1);

    Eigen::MatrixXd basis1 = corr1.current().basis();

    auto pred2 = filter.predict(1.0, corr1);
    Eigen::VectorXd meas2(2);
    meas2 << 2.1, 0.05;
    auto [corr2, lik2] = filter.correct(meas2, pred2);

    Eigen::MatrixXd basis2 = corr2.current().basis();

    Eigen::MatrixXd alignment = (basis2.transpose() * basis1).cwiseAbs();
    for (int k = 0; k < 2; ++k) {
        EXPECT_GT(alignment(k, k), 0.5);
    }

    EXPECT_EQ(corr2.window_size(), 3);
    EXPECT_EQ(corr2.stacked_size(), 12);
}

TEST_F(TrajectoryGGIWOrientationEKFTest, Gate) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation<>tg(4, models::GGIWOrientation<>(10.0, 5.0, mean, cov, 10.0, V), kTestWindow);

    Eigen::VectorXd meas_close(2);
    meas_close << 0.1, 0.1;
    double gate_close = filter.gate(meas_close, tg);
    EXPECT_LT(gate_close, 1.0);

    Eigen::VectorXd meas_far(2);
    meas_far << 100.0, 100.0;
    double gate_far = filter.gate(meas_far, tg);
    EXPECT_GT(gate_far, 100.0);
}
