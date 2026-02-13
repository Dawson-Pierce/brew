#include <gtest/gtest.h>
#include "brew/multi_target/elementary_symmetric.hpp"
#include "brew/multi_target/cphd.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"
#include <random>

using namespace brew;

// ---- Elementary Symmetric Function tests ----

TEST(ElementarySymmetric, EmptyVector) {
    Eigen::VectorXd z(0);
    auto esf = multi_target::elementary_symmetric_functions(z);
    ASSERT_EQ(esf.size(), 1);
    EXPECT_DOUBLE_EQ(esf(0), 1.0);
}

TEST(ElementarySymmetric, SingleElement) {
    Eigen::VectorXd z(1);
    z << 3.0;
    auto esf = multi_target::elementary_symmetric_functions(z);
    ASSERT_EQ(esf.size(), 2);
    EXPECT_DOUBLE_EQ(esf(0), 1.0);
    EXPECT_DOUBLE_EQ(esf(1), 3.0);
}

TEST(ElementarySymmetric, ThreeElements) {
    // z = [1, 2, 3]
    // e_0 = 1
    // e_1 = 1+2+3 = 6
    // e_2 = 1*2 + 1*3 + 2*3 = 11
    // e_3 = 1*2*3 = 6
    Eigen::VectorXd z(3);
    z << 1.0, 2.0, 3.0;
    auto esf = multi_target::elementary_symmetric_functions(z);
    ASSERT_EQ(esf.size(), 4);
    EXPECT_DOUBLE_EQ(esf(0), 1.0);
    EXPECT_DOUBLE_EQ(esf(1), 6.0);
    EXPECT_DOUBLE_EQ(esf(2), 11.0);
    EXPECT_DOUBLE_EQ(esf(3), 6.0);
}

TEST(ElementarySymmetric, ESFExcluding) {
    // z = [1, 2, 3], exclude z_1 = 2 => z' = [1, 3]
    // esf(z') = [1, 4, 3]
    Eigen::VectorXd z(3);
    z << 1.0, 2.0, 3.0;
    auto esf_full = multi_target::elementary_symmetric_functions(z);
    auto esf_minus = multi_target::esf_excluding(esf_full, 2.0);

    ASSERT_EQ(esf_minus.size(), 3);
    EXPECT_DOUBLE_EQ(esf_minus(0), 1.0);
    EXPECT_DOUBLE_EQ(esf_minus(1), 4.0);  // 1 + 3
    EXPECT_DOUBLE_EQ(esf_minus(2), 3.0);  // 1*3
}

TEST(ElementarySymmetric, FallingFactorial) {
    EXPECT_DOUBLE_EQ(multi_target::falling_factorial(5, 0), 1.0);
    EXPECT_DOUBLE_EQ(multi_target::falling_factorial(5, 1), 5.0);
    EXPECT_DOUBLE_EQ(multi_target::falling_factorial(5, 2), 20.0);  // 5*4
    EXPECT_DOUBLE_EQ(multi_target::falling_factorial(5, 3), 60.0);  // 5*4*3
    EXPECT_DOUBLE_EQ(multi_target::falling_factorial(5, 5), 120.0); // 5!
}

// ---- CPHD filter tests ----

TEST(CPHD, GaussianPredictCorrectCleanup) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.1 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::VectorXd birth_mean(4);
    birth_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd birth_cov = 10.0 * Eigen::MatrixXd::Identity(4, 4);
    birth->add_component(std::make_unique<distributions::Gaussian>(birth_mean, birth_cov), 0.1);

    auto intensity = birth->clone();

    multi_target::CPHD<distributions::Gaussian> cphd;
    cphd.set_filter(std::move(ekf));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::move(intensity));
    cphd.set_prob_detection(0.9);
    cphd.set_prob_survive(0.99);
    cphd.set_clutter_rate(1.0);
    cphd.set_clutter_density(1e-4);
    cphd.set_prune_threshold(1e-5);
    cphd.set_merge_threshold(4.0);
    cphd.set_max_components(20);
    cphd.set_extract_threshold(0.4);
    cphd.set_gate_threshold(16.0);

    // Initialize cardinality as Poisson(0.1) and birth as Poisson(0.1)
    cphd.set_poisson_cardinality(0.1);
    cphd.set_poisson_birth_cardinality(0.1);

    // Predict
    cphd.predict(0, 1.0);
    EXPECT_GE(cphd.intensity().size(), 2u);

    // Cardinality should be valid PMF
    EXPECT_NEAR(cphd.cardinality().sum(), 1.0, 1e-10);

    // Correct with one measurement near origin
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    cphd.correct(meas);
    EXPECT_GE(cphd.intensity().size(), 1u);
    EXPECT_NEAR(cphd.cardinality().sum(), 1.0, 1e-10);

    // Cleanup
    cphd.cleanup();
    EXPECT_LE(cphd.intensity().size(), 20u);
    EXPECT_GE(cphd.extracted_mixtures().size(), 1u);
    EXPECT_GE(cphd.cardinality_history().size(), 1u);
}

TEST(CPHD, Clone) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(Eigen::MatrixXd::Identity(2, 2));
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);

    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::VectorXd m(4);
    m.setZero();
    birth->add_component(std::make_unique<distributions::Gaussian>(m, Eigen::MatrixXd::Identity(4, 4)), 0.1);

    multi_target::CPHD<distributions::Gaussian> cphd;
    cphd.set_filter(std::move(ekf));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::make_unique<distributions::Mixture<distributions::Gaussian>>());
    cphd.set_poisson_cardinality(0.5);
    cphd.set_poisson_birth_cardinality(0.1);

    auto cloned = cphd.clone();
    ASSERT_NE(cloned, nullptr);
}

TEST(CPHD, CardinalityConvergence) {
    // Run a few steps with consistent 2-target measurements.
    // Cardinality should converge toward 2.
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 100.0; birth_cov(1, 1) = 100.0;
    birth_cov(2, 2) = 10.0; birth_cov(3, 3) = 10.0;
    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.05);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.05);

    multi_target::CPHD<distributions::Gaussian> cphd;
    cphd.set_filter(std::move(ekf));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::make_unique<distributions::Mixture<distributions::Gaussian>>());
    cphd.set_prob_detection(0.95);
    cphd.set_prob_survive(0.99);
    cphd.set_clutter_rate(1.0);
    cphd.set_clutter_density(1e-4);
    cphd.set_prune_threshold(1e-5);
    cphd.set_merge_threshold(4.0);
    cphd.set_max_components(30);
    cphd.set_extract_threshold(0.4);
    cphd.set_gate_threshold(25.0);
    cphd.set_poisson_cardinality(0.1);
    cphd.set_poisson_birth_cardinality(0.1);

    // Simulate two stationary targets at (5,5) and (50,5) for 20 steps
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    for (int k = 0; k < 20; ++k) {
        cphd.predict(k, 1.0);

        // Two measurements, one near each target
        Eigen::MatrixXd meas(2, 2);
        meas(0, 0) = 5.0 + noise(rng);
        meas(1, 0) = 5.0 + noise(rng);
        meas(0, 1) = 50.0 + noise(rng);
        meas(1, 1) = 5.0 + noise(rng);

        cphd.correct(meas);
        cphd.cleanup();
    }

    // After 20 steps with consistent 2 measurements, estimated cardinality should be near 2
    double est = cphd.estimated_cardinality();
    EXPECT_GT(est, 1.0) << "Estimated cardinality should be above 1 (two targets present)";
    EXPECT_LT(est, 4.0) << "Estimated cardinality should not overshoot too much";
}
