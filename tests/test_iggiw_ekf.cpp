#include <gtest/gtest.h>
#include "brew/iggiw/filters/iggiw_ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"

using namespace brew;

class IGGIWEKFTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 4-state 2D integrator: [x, y, vx, vy]
        auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
        filter.set_dynamics(dyn);

        // Position-only measurement H = [I2 0]
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        filter.set_measurement_jacobian(H);

        filter.set_process_noise(0.1 * Eigen::MatrixXd::Identity(4, 4));
        filter.set_measurement_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
    }

    filters::IGGIWEKF<> filter;
};

TEST_F(IGGIWEKFTest, PredictDecay) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * (2.0 * 2 + 2.0 + 2.0) * Eigen::MatrixXd::Identity(2, 2);

    // alpha > 1 required so the InverseGamma mean beta/(alpha-1) is finite.
    models::IGGIW<> iggiw(10.0, 5.0, mean, cov, 10.0, V);

    // delta = 1 → no decay; gamma_growth = 1 → no drift in mean intensity.
    filter.set_intensity_forgetting_factor(1.0);
    filter.set_intensity_growth(1.0);
    filter.set_extent_forgetting_factor(1.0);

    auto pred = filter.predict(1.0, iggiw);

    // Mean propagates: x += vx*dt
    EXPECT_NEAR(pred.mean()(0), 1.0, 1e-10);
    EXPECT_NEAR(pred.mean()(1), 0.0, 1e-10);

    // Covariance grows
    EXPECT_GT(pred.covariance()(0, 0), cov(0, 0));

    // With delta_gamma=1 and gamma_growth=1, the gamma parameters are preserved
    // (next_alpha = 1 + 1*(alpha-1) = alpha; next_beta = gamma_bar*(next_alpha-1) = beta).
    EXPECT_NEAR(pred.alpha(), 10.0, 1e-10);
    EXPECT_NEAR(pred.beta(), 5.0, 1e-10);

    // IW dof preserved at delta_extent = 1
    EXPECT_NEAR(pred.v(), 10.0, 1e-10);
}

TEST_F(IGGIWEKFTest, PredictDecayShrinksToward1) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 50.0 * Eigen::MatrixXd::Identity(2, 2);

    models::IGGIW<> iggiw(10.0, 9.0, mean, cov, 10.0, V);

    filter.set_intensity_forgetting_factor(0.5);
    filter.set_intensity_growth(1.0);
    filter.set_extent_forgetting_factor(0.5);

    auto pred = filter.predict(1.0, iggiw);

    // delta_gamma < 1 ⇒ alpha shrinks toward 1
    EXPECT_LT(pred.alpha(), iggiw.alpha());
    EXPECT_GT(pred.alpha(), 1.0);

    // Mean intensity beta/(alpha-1) preserved when gamma_growth = 1
    const double m0 = iggiw.beta() / (iggiw.alpha() - 1.0);
    const double m1 = pred.beta() / (pred.alpha() - 1.0);
    EXPECT_NEAR(m0, m1, 1e-10);

    // delta_extent < 1 ⇒ v shrinks toward floor 2d+2 = 6
    EXPECT_LT(pred.v(), iggiw.v());
    EXPECT_GT(pred.v(), 6.0);
}

TEST_F(IGGIWEKFTest, CorrectWeightedMeasurements) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 50.0 * Eigen::MatrixXd::Identity(2, 2);

    // v large enough to keep the predictive extent well-conditioned with omega=5
    models::IGGIW<> iggiw(10.0, 5.0, mean, cov, 20.0, V);
    filter.set_eta(1.0);
    filter.set_omega(5.0);
    filter.set_lambda(4.0);

    // Three weighted measurements, packed as (d+1)*N = 3*3 = 9-vector:
    // [x0,y0,w0, x1,y1,w1, x2,y2,w2]
    Eigen::VectorXd meas(9);
    meas << 0.1, 0.1, 2.0,
            0.2, 0.0, 1.0,
            0.0, 0.2, 1.0;

    auto [corrected, likelihood] = filter.correct(meas, iggiw);

    // Mean drifts toward the weighted centroid (positive in both dims)
    EXPECT_GT(corrected.mean()(0), 0.0);
    EXPECT_GT(corrected.mean()(1), 0.0);

    // Position covariance shrinks after a correction
    EXPECT_LT(corrected.covariance()(0, 0), cov(0, 0));

    // Likelihood positive and finite
    EXPECT_GT(likelihood, 0.0);
    EXPECT_TRUE(std::isfinite(likelihood));

    // Conjugate update: alpha increases by eta, beta increases by eta * mean(weights).
    // mean(weights) = (2 + 1 + 1)/3 = 4/3
    EXPECT_NEAR(corrected.alpha(), 10.0 + 1.0, 1e-10);
    EXPECT_NEAR(corrected.beta(),  5.0 + 1.0 * (4.0 / 3.0), 1e-10);

    // IW dof increases by omega
    EXPECT_NEAR(corrected.v(), 20.0 + 5.0, 1e-10);
}

TEST_F(IGGIWEKFTest, CorrectSingleUnweighted) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 50.0 * Eigen::MatrixXd::Identity(2, 2);

    models::IGGIW<> iggiw(10.0, 5.0, mean, cov, 10.0, V);

    // Length-d fallback path: single measurement, weight defaults to 1.
    Eigen::VectorXd meas(2);
    meas << 0.1, 0.1;

    auto [corrected, likelihood] = filter.correct(meas, iggiw);
    EXPECT_GT(likelihood, 0.0);
    EXPECT_TRUE(std::isfinite(likelihood));
    EXPECT_GT(corrected.mean()(0), 0.0);
}

TEST_F(IGGIWEKFTest, CorrectRejectsNonPositiveWeights) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 50.0 * Eigen::MatrixXd::Identity(2, 2);

    models::IGGIW<> iggiw(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::VectorXd meas(3);
    meas << 0.1, 0.1, 0.0;  // zero weight: invalid

    EXPECT_THROW(filter.correct(meas, iggiw), std::invalid_argument);
}

TEST_F(IGGIWEKFTest, Gate) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    models::IGGIW<> iggiw(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::VectorXd meas_close(3);
    meas_close << 0.1, 0.1, 1.0;
    double gate_close = filter.gate(meas_close, iggiw);
    EXPECT_LT(gate_close, 1.0);

    Eigen::VectorXd meas_far(3);
    meas_far << 100.0, 100.0, 1.0;
    double gate_far = filter.gate(meas_far, iggiw);
    EXPECT_GT(gate_far, 100.0);
}

TEST_F(IGGIWEKFTest, SetterRejectsBadValues) {
    EXPECT_THROW(filter.set_eta(0.0), std::invalid_argument);
    EXPECT_THROW(filter.set_lambda(-1.0), std::invalid_argument);
    EXPECT_THROW(filter.set_omega(0.0), std::invalid_argument);
    EXPECT_THROW(filter.set_intensity_forgetting_factor(0.0), std::invalid_argument);
    EXPECT_THROW(filter.set_intensity_forgetting_factor(1.5), std::invalid_argument);
    EXPECT_THROW(filter.set_extent_forgetting_factor(2.0), std::invalid_argument);
    EXPECT_THROW(filter.set_intensity_growth(-0.5), std::invalid_argument);
}
