#include <gtest/gtest.h>
#include "brew/multi_target/phd.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

TEST(PHD, GaussianPredictCorrectCleanup) {
    // Setup filter
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.1 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    // Birth model: one component at origin
    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::VectorXd birth_mean(4);
    birth_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd birth_cov = 10.0 * Eigen::MatrixXd::Identity(4, 4);
    birth->add_component(std::make_unique<models::Gaussian>(birth_mean, birth_cov), 0.1);

    // Initial intensity = copy of birth
    auto intensity = birth->clone();

    // PHD setup
    multi_target::PHD<models::Gaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(0.9);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(20);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(16.0);

    // Predict
    phd.predict(0, 1.0);
    EXPECT_GE(phd.intensity().size(), 2u); // survived + birth

    // Correct with one measurement near origin
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    phd.correct(meas);
    EXPECT_GE(phd.intensity().size(), 1u);

    // Cleanup (prune + merge + cap + extract)
    phd.cleanup();
    EXPECT_LE(phd.intensity().size(), 20u);
    EXPECT_GE(phd.extracted_mixtures().size(), 1u);
}

TEST(PHD, Clone) {
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

    multi_target::PHD<models::Gaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::Gaussian>>());

    auto cloned = phd.clone();
    ASSERT_NE(cloned, nullptr);
}
