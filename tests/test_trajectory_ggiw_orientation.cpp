#include <gtest/gtest.h>
#include "brew/filters/trajectory_ggiw_orientation_ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"
#include "brew/serialization/rfs_json.hpp"

using namespace brew;

// ============================================================
// Model tests
// ============================================================

TEST(TrajectoryGGIWOrientationModel, ConstructAndAccessors) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 10.0, 2.0,
          2.0, 5.0;

    models::TrajectoryGGIWOrientation g(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    EXPECT_EQ(g.extent_dim(), 2);
    EXPECT_TRUE(g.is_extended());
    EXPECT_DOUBLE_EQ(g.alpha(), 10.0);
    EXPECT_DOUBLE_EQ(g.beta(), 5.0);
    EXPECT_DOUBLE_EQ(g.v(), 10.0);
    EXPECT_EQ(g.state_dim, 4);
    EXPECT_EQ(g.init_idx, 0);
    EXPECT_EQ(g.window_size, 1);

    // Basis should be initialized from V decomposition
    EXPECT_EQ(g.basis().rows(), 2);
    EXPECT_EQ(g.basis().cols(), 2);
    EXPECT_TRUE(g.has_eigenvalues());

    // Trajectory helpers
    EXPECT_EQ(g.get_last_state().size(), 4);
    EXPECT_EQ(g.get_last_cov().rows(), 4);
    EXPECT_EQ(g.mean_history().cols(), 1);
}

TEST(TrajectoryGGIWOrientationModel, Clone) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation g(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    auto typed_clone = g.clone_typed();
    ASSERT_NE(typed_clone, nullptr);
    EXPECT_DOUBLE_EQ(typed_clone->alpha(), 10.0);
    EXPECT_EQ(typed_clone->basis().rows(), 2);
    EXPECT_EQ(typed_clone->state_dim, 4);
    EXPECT_TRUE(typed_clone->has_eigenvalues());
}

// ============================================================
// Filter tests
// ============================================================

class TrajectoryGGIWOrientationEKFTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto dyn = std::make_shared<dynamics::Integrator2D>();
        filter.set_dynamics(dyn);
        filter.set_window_size(10);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        filter.set_measurement_jacobian(H);

        filter.set_process_noise(0.1 * Eigen::MatrixXd::Identity(4, 4));
        filter.set_measurement_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
        filter.set_temporal_decay(1.0);
        filter.set_forgetting_factor(1.0);
    }

    filters::TrajectoryGGIWOrientationEKF filter;
};

TEST_F(TrajectoryGGIWOrientationEKFTest, PredictGrowsTrajectory) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation tg(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    auto pred = filter.predict(1.0, tg);

    // Window should grow
    EXPECT_EQ(pred.window_size, 2);
    EXPECT_EQ(pred.mean().size(), 8);  // 2 time steps * 4 state dim

    // Last state should propagate
    Eigen::VectorXd last = pred.get_last_state();
    EXPECT_NEAR(last(0), 1.0, 1e-10);  // x += vx*dt

    // Basis should be preserved
    EXPECT_EQ(pred.basis().rows(), 2);
}

TEST_F(TrajectoryGGIWOrientationEKFTest, CorrectPopulatesBasis) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 50.0, 10.0,
         10.0, 30.0;

    models::TrajectoryGGIWOrientation tg(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    // Predict then correct
    auto pred = filter.predict(1.0, tg);

    Eigen::VectorXd meas(2);
    meas << 1.1, 0.1;
    auto [corrected, likelihood] = filter.correct(meas, pred);

    EXPECT_GT(likelihood, 0.0);
    // Alpha increases by W=1 from predicted (which was decayed from 10 by eta=1 -> still 10)
    EXPECT_DOUBLE_EQ(corrected.alpha(), pred.alpha() + 1.0);
    EXPECT_DOUBLE_EQ(corrected.beta(), pred.beta() + 1.0);
    // v increases by W=1 from predicted (which was decayed during predict)
    EXPECT_DOUBLE_EQ(corrected.v(), pred.v() + 1.0);

    // Basis should be populated from SVD alignment
    EXPECT_EQ(corrected.basis().rows(), 2);
    EXPECT_EQ(corrected.basis().cols(), 2);
    EXPECT_TRUE(corrected.has_eigenvalues());

    // Window size unchanged by correct
    EXPECT_EQ(corrected.window_size, pred.window_size);
}

TEST_F(TrajectoryGGIWOrientationEKFTest, MultipleStepsBasisAlignment) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 80.0, 5.0,
          5.0, 20.0;

    models::TrajectoryGGIWOrientation tg(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    // Step 1: predict + correct
    auto pred1 = filter.predict(1.0, tg);
    Eigen::VectorXd meas1(2);
    meas1 << 1.05, 0.05;
    auto [corr1, lik1] = filter.correct(meas1, pred1);

    Eigen::MatrixXd basis1 = corr1.basis();

    // Step 2: predict + correct
    auto pred2 = filter.predict(1.0, corr1);
    Eigen::VectorXd meas2(2);
    meas2 << 2.1, 0.05;
    auto [corr2, lik2] = filter.correct(meas2, pred2);

    Eigen::MatrixXd basis2 = corr2.basis();

    // Basis should remain aligned across steps
    Eigen::MatrixXd alignment = (basis2.transpose() * basis1).cwiseAbs();
    for (int k = 0; k < 2; ++k) {
        EXPECT_GT(alignment(k, k), 0.5);
    }

    // Trajectory should have grown
    EXPECT_EQ(corr2.window_size, 3);
    EXPECT_EQ(corr2.mean().size(), 12);  // 3 steps * 4
}

TEST_F(TrajectoryGGIWOrientationEKFTest, Gate) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation tg(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    Eigen::VectorXd meas_close(2);
    meas_close << 0.1, 0.1;
    double gate_close = filter.gate(meas_close, tg);
    EXPECT_LT(gate_close, 1.0);

    Eigen::VectorXd meas_far(2);
    meas_far << 100.0, 100.0;
    double gate_far = filter.gate(meas_far, tg);
    EXPECT_GT(gate_far, 100.0);
}

// ============================================================
// Serialization tests
// ============================================================

TEST(TrajectoryGGIWOrientationSerialization, RoundTrip) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = 2.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 10.0, 2.0,
          2.0, 5.0;

    models::TrajectoryGGIWOrientation original(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    auto j = serialization::to_json(original);
    EXPECT_EQ(j["type"], "TrajectoryGGIWOrientation");

    auto restored = serialization::trajectory_ggiw_orientation_from_json(j);

    EXPECT_DOUBLE_EQ(restored.alpha(), original.alpha());
    EXPECT_DOUBLE_EQ(restored.beta(), original.beta());
    EXPECT_DOUBLE_EQ(restored.v(), original.v());
    EXPECT_EQ(restored.state_dim, 4);
    EXPECT_EQ(restored.init_idx, 0);

    for (int i = 0; i < mean.size(); ++i) {
        EXPECT_NEAR(restored.mean()(i), original.mean()(i), 1e-10);
    }

    // Basis should have been re-derived from V
    EXPECT_EQ(restored.basis().rows(), 2);
    EXPECT_TRUE(restored.has_eigenvalues());
}

TEST(TrajectoryGGIWOrientationSerialization, DistributionSerializer) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::TrajectoryGGIWOrientation original(0, 4, mean, cov, 10.0, 5.0, 10.0, V);

    auto j = serialization::DistributionSerializer<models::TrajectoryGGIWOrientation>::serialize(original);
    auto restored = serialization::DistributionSerializer<models::TrajectoryGGIWOrientation>::deserialize(j);

    EXPECT_DOUBLE_EQ(restored.alpha(), 10.0);
    EXPECT_EQ(restored.state_dim, 4);
}
