#include <gtest/gtest.h>
#include "brew/trajectory_gaussian/filters/trajectory_gaussian_ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"

using namespace brew;

class TrajectoryGaussianEKFTest : public ::testing::Test {
protected:
    static constexpr int kWindow = 5;

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

    filters::TrajectoryGaussianEKF<kWindow> filter;
};

TEST_F(TrajectoryGaussianEKFTest, PredictWindowGrowth) {
    // Initial single-state trajectory
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    models::TrajectoryGaussian<kWindow> tg(4, models::Gaussian<>(mean, cov));
    EXPECT_EQ(tg.window_size(), 1);

    // Predict should grow the live window (mean() is the full fixed-capacity
    // stacked vector; stacked_size() tracks the live portion).
    auto pred1 = filter.predict(1.0, tg);
    EXPECT_EQ(pred1.window_size(), 2);
    EXPECT_EQ(pred1.stacked_size(), 8); // 2 states * 4 dims

    // Mean of new state should be propagated
    EXPECT_NEAR(pred1.get_last_state()(0), 1.0, 1e-10);
    EXPECT_NEAR(pred1.get_last_state()(2), 1.0, 1e-10); // velocity preserved

    // Second predict
    auto pred2 = filter.predict(1.0, pred1);
    EXPECT_EQ(pred2.window_size(), 3);
    EXPECT_EQ(pred2.stacked_size(), 12);
}

TEST_F(TrajectoryGaussianEKFTest, LScanTrim) {
    // Build a trajectory and predict past the window so the ring buffer slides.
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    models::TrajectoryGaussian<kWindow> tg(4, models::Gaussian<>(mean, cov));

    // l_window_ is 5. Predict 6 times starting from window=1; after slot 5 is full,
    // further predicts slide the buffer left (oldest dropped), so window stays at 5.
    auto current = tg;
    for (int i = 0; i < 6; ++i) {
        current = filter.predict(1.0, current);
    }

    // History and stacked state are both capped at l_window.
    EXPECT_EQ(current.window_size(), 5);
    EXPECT_EQ(current.mean().size(), 5 * 4);
    EXPECT_EQ(current.covariance().rows(), 5 * 4);
    EXPECT_EQ(current.covariance().cols(), 5 * 4);
}

TEST_F(TrajectoryGaussianEKFTest, CorrectStep) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    models::TrajectoryGaussian<kWindow> tg(4, models::Gaussian<>(mean, cov));

    auto pred = filter.predict(1.0, tg);

    Eigen::VectorXd meas(2);
    meas << 1.1, 0.1;

    auto [corrected, likelihood] = filter.correct(meas, pred);

    // Window size should stay the same after correct
    EXPECT_EQ(corrected.window_size(), pred.window_size());

    // Last state should move toward measurement
    EXPECT_GT(corrected.get_last_state()(0), 0.9);

    // Likelihood should be positive
    EXPECT_GT(likelihood, 0.0);
}

TEST_F(TrajectoryGaussianEKFTest, Gate) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    models::TrajectoryGaussian<kWindow> tg(4, models::Gaussian<>(mean, cov));

    Eigen::VectorXd meas_close(2);
    meas_close << 0.1, 0.1;
    double gate_close = filter.gate(meas_close, tg);
    EXPECT_LT(gate_close, 1.0);

    Eigen::VectorXd meas_far(2);
    meas_far << 100.0, 100.0;
    double gate_far = filter.gate(meas_far, tg);
    EXPECT_GT(gate_far, 100.0);
}
