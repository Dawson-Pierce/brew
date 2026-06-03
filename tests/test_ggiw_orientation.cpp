#include <gtest/gtest.h>
#include "brew/ggiw_orientation/filters/ggiw_orientation_ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"

using namespace brew;

TEST(GGIWOrientationModel, ConstructAndAccessors) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    Eigen::MatrixXd V(2, 2);
    V << 10.0, 2.0,
          2.0, 5.0;

    models::GGIWOrientation<> g(10.0, 5.0, mean, cov, 10.0, V);

    EXPECT_EQ(g.extent_dim(), 2);
    EXPECT_TRUE(g.is_extended());
    EXPECT_DOUBLE_EQ(g.alpha(), 10.0);
    EXPECT_DOUBLE_EQ(g.beta(), 5.0);
    EXPECT_DOUBLE_EQ(g.v(), 10.0);
    EXPECT_EQ(g.mean().size(), 4);

    EXPECT_EQ(g.basis().rows(), 2);
    EXPECT_EQ(g.basis().cols(), 2);
    EXPECT_TRUE(g.has_eigenvalues());
    EXPECT_EQ(g.eigenvalues().rows(), 2);
    EXPECT_EQ(g.eigenvalues().cols(), 2);

    Eigen::MatrixXd I_check = g.basis().transpose() * g.basis();
    EXPECT_NEAR(I_check(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(I_check(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(I_check(0, 1), 0.0, 1e-10);
}

TEST(GGIWOrientationModel, Clone) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 0.5, -0.5;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIWOrientation<> g(10.0, 5.0, mean, cov, 10.0, V);

    auto base_clone = g.clone();
    ASSERT_NE(base_clone, nullptr);

    auto typed_clone = g.clone_typed();
    ASSERT_NE(typed_clone, nullptr);
    EXPECT_DOUBLE_EQ(typed_clone->alpha(), 10.0);
    EXPECT_EQ(typed_clone->basis().rows(), 2);
    EXPECT_TRUE(typed_clone->has_eigenvalues());
}

TEST(GGIWOrientationModel, SetBasis) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIWOrientation<> g(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::MatrixXd new_basis(2, 2);
    new_basis << 0.0, 1.0,
                 1.0, 0.0;
    g.set_basis(new_basis);

    EXPECT_NEAR(g.basis()(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(g.basis()(0, 1), 1.0, 1e-10);
}

class GGIWOrientationEKFTest : public ::testing::Test {
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

    filters::GGIWOrientationEKF<> filter;
};

TEST_F(GGIWOrientationEKFTest, PredictPreservesBasis) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIWOrientation<> ggiw(10.0, 5.0, mean, cov, 10.0, V);
    filter.set_temporal_decay(1.0);
    filter.set_forgetting_factor(1.0);

    auto pred = filter.predict(1.0, ggiw);

    EXPECT_NEAR(pred.mean()(0), 1.0, 1e-10);

    EXPECT_EQ(pred.basis().rows(), 2);
    EXPECT_EQ(pred.basis().cols(), 2);
}

TEST_F(GGIWOrientationEKFTest, FirstCorrectionPopulatesBasis) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    Eigen::MatrixXd V(2, 2);
    V << 50.0, 10.0,
         10.0, 30.0;

    models::GGIWOrientation<> ggiw(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::VectorXd meas(2);
    meas << 0.1, 0.1;

    auto [corrected, likelihood] = filter.correct(meas, ggiw);

    EXPECT_EQ(corrected.basis().rows(), 2);
    EXPECT_EQ(corrected.basis().cols(), 2);
    EXPECT_TRUE(corrected.has_eigenvalues());

    EXPECT_GT(likelihood, 0.0);

    EXPECT_DOUBLE_EQ(corrected.alpha(), 11.0);
    EXPECT_DOUBLE_EQ(corrected.beta(), 6.0);
    EXPECT_DOUBLE_EQ(corrected.v(), 11.0);
}

TEST_F(GGIWOrientationEKFTest, SecondCorrectionAlignsBasis) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    Eigen::MatrixXd V(2, 2);
    V << 80.0, 5.0,
          5.0, 20.0;

    models::GGIWOrientation<> ggiw(10.0, 5.0, mean, cov, 10.0, V);
    filter.set_temporal_decay(1.0);
    filter.set_forgetting_factor(1.0);

    Eigen::VectorXd meas1(2);
    meas1 << 0.05, 0.05;
    auto [corr1, lik1] = filter.correct(meas1, ggiw);

    Eigen::MatrixXd basis1 = corr1.basis();
    EXPECT_EQ(basis1.rows(), 2);

    auto pred = filter.predict(1.0, corr1);
    Eigen::VectorXd meas2(2);
    meas2 << 1.1, 0.05;
    auto [corr2, lik2] = filter.correct(meas2, pred);

    Eigen::MatrixXd basis2 = corr2.basis();

    Eigen::MatrixXd alignment = (basis2.transpose() * basis1).cwiseAbs();
    for (int k = 0; k < 2; ++k) {

        EXPECT_GT(alignment(k, k), 0.5);
    }
}

TEST_F(GGIWOrientationEKFTest, Gate) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIWOrientation<> ggiw(10.0, 5.0, mean, cov, 10.0, V);

    Eigen::VectorXd meas_close(2);
    meas_close << 0.1, 0.1;
    double gate_close = filter.gate(meas_close, ggiw);
    EXPECT_LT(gate_close, 1.0);

    Eigen::VectorXd meas_far(2);
    meas_far << 100.0, 100.0;
    double gate_far = filter.gate(meas_far, ggiw);
    EXPECT_GT(gate_far, 100.0);
}
