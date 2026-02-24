#include <gtest/gtest.h>
#include "brew/multi_target/mbm.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

TEST(MBM, GaussianPredictCorrectCleanup) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.1 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    // Birth model
    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::VectorXd birth_mean(4);
    birth_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd birth_cov = 10.0 * Eigen::MatrixXd::Identity(4, 4);
    birth->add_component(std::make_unique<models::Gaussian>(birth_mean, birth_cov), 0.1);

    multi_target::MBM<models::Gaussian> mbm;
    mbm.set_filter(std::move(ekf));
    mbm.set_birth_model(std::move(birth));
    mbm.set_prob_detection(0.9);
    mbm.set_prob_survive(0.99);
    mbm.set_clutter_rate(1.0);
    mbm.set_clutter_density(1e-4);
    mbm.set_prune_threshold_hypothesis(1e-4);
    mbm.set_prune_threshold_bernoulli(1e-3);
    mbm.set_max_hypotheses(20);
    mbm.set_extract_threshold(0.4);
    mbm.set_gate_threshold(16.0);
    mbm.set_k_best(5);

    // Predict
    mbm.predict(0, 1.0);
    EXPECT_GE(mbm.global_hypotheses().size(), 1u);

    // Correct with one measurement near origin
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    mbm.correct(meas);
    EXPECT_GE(mbm.global_hypotheses().size(), 1u);

    // Cleanup
    mbm.cleanup();
    EXPECT_GE(mbm.extracted_mixtures().size(), 1u);
}

TEST(MBM, Clone) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(Eigen::MatrixXd::Identity(2, 2));
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);

    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::VectorXd m(4);
    m.setZero();
    birth->add_component(std::make_unique<models::Gaussian>(m, Eigen::MatrixXd::Identity(4, 4)), 0.1);

    multi_target::MBM<models::Gaussian> mbm;
    mbm.set_filter(std::move(ekf));
    mbm.set_birth_model(std::move(birth));

    auto cloned = mbm.clone();
    ASSERT_NE(cloned, nullptr);
}

TEST(MBM, MultipleHypothesesGenerated) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::MatrixXd birth_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(std::make_unique<models::Gaussian>(b1, birth_cov), 0.1);
    birth->add_component(std::make_unique<models::Gaussian>(b2, birth_cov), 0.1);

    multi_target::MBM<models::Gaussian> mbm;
    mbm.set_filter(std::move(ekf));
    mbm.set_birth_model(std::move(birth));
    mbm.set_prob_detection(0.9);
    mbm.set_prob_survive(0.99);
    mbm.set_clutter_rate(1.0);
    mbm.set_clutter_density(1e-4);
    mbm.set_gate_threshold(25.0);
    mbm.set_k_best(5);
    mbm.set_max_hypotheses(50);
    mbm.set_extract_threshold(0.4);

    mbm.predict(0, 1.0);

    // Two measurements: should generate multiple hypotheses (different associations)
    Eigen::MatrixXd meas(2, 2);
    meas << 1.0, 49.0,
            1.0, 1.0;
    mbm.correct(meas);

    // With 2 Bernoullis and 2 measurements, k_best=5 should produce multiple hypotheses
    EXPECT_GE(mbm.global_hypotheses().size(), 2u);
}
