#include <gtest/gtest.h>
#include "brew/multi_target/pmbm.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

TEST(PMBM, GaussianPredictCorrectCleanup) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.1 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    // Birth model (added to Poisson each predict step)
    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::VectorXd birth_mean(4);
    birth_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd birth_cov = 10.0 * Eigen::MatrixXd::Identity(4, 4);
    birth->add_component(std::make_unique<models::Gaussian>(birth_mean, birth_cov), 0.1);

    // Initial Poisson intensity
    auto poisson = std::make_unique<models::Mixture<models::Gaussian>>();

    multi_target::PMBM<models::Gaussian> pmbm;
    pmbm.set_filter(std::move(ekf));
    pmbm.set_birth_model(std::move(birth));
    pmbm.set_poisson_intensity(std::move(poisson));
    pmbm.set_prob_detection(0.9);
    pmbm.set_prob_survive(0.99);
    pmbm.set_clutter_rate(1.0);
    pmbm.set_clutter_density(1e-4);
    pmbm.set_prune_threshold_hypothesis(1e-4);
    pmbm.set_prune_threshold_bernoulli(1e-3);
    pmbm.set_max_hypotheses(20);
    pmbm.set_extract_threshold(0.4);
    pmbm.set_gate_threshold(16.0);
    pmbm.set_k_best(5);

    // Predict
    pmbm.predict(0, 1.0);
    EXPECT_GE(pmbm.poisson_intensity().size(), 1u)
        << "Poisson should have birth components after predict";

    // Correct with one measurement near origin
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    pmbm.correct(meas);

    // Cleanup
    pmbm.cleanup();
    EXPECT_GE(pmbm.extracted_mixtures().size(), 1u);

    // Cardinality estimation should be available after cleanup
    EXPECT_GE(pmbm.cardinality().size(), 1);
    EXPECT_GE(pmbm.estimated_cardinality(), 0.0);
    // PMF should sum to ~1
    EXPECT_NEAR(pmbm.cardinality().sum(), 1.0, 1e-6);
}

TEST(PMBM, Clone) {
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

    auto poisson = std::make_unique<models::Mixture<models::Gaussian>>();

    multi_target::PMBM<models::Gaussian> pmbm;
    pmbm.set_filter(std::move(ekf));
    pmbm.set_birth_model(std::move(birth));
    pmbm.set_poisson_intensity(std::move(poisson));

    auto cloned = pmbm.clone();
    ASSERT_NE(cloned, nullptr);
}

TEST(PMBM, PoissonSpawnsNewTracks) {
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

    auto poisson = std::make_unique<models::Mixture<models::Gaussian>>();

    multi_target::PMBM<models::Gaussian> pmbm;
    pmbm.set_filter(std::move(ekf));
    pmbm.set_birth_model(std::move(birth));
    pmbm.set_poisson_intensity(std::move(poisson));
    pmbm.set_prob_detection(0.9);
    pmbm.set_prob_survive(0.99);
    pmbm.set_clutter_rate(1.0);
    pmbm.set_clutter_density(1e-4);
    pmbm.set_gate_threshold(25.0);
    pmbm.set_k_best(5);
    pmbm.set_max_hypotheses(50);
    pmbm.set_extract_threshold(0.4);

    pmbm.predict(0, 1.0);
    EXPECT_EQ(pmbm.num_tracks(), 0) << "No Bernoulli tracks before first correction";

    // Two measurements near both birth locations
    Eigen::MatrixXd meas(2, 2);
    meas << 1.0, 49.0,
            1.0, 1.0;
    pmbm.correct(meas);

    // Should have spawned new Bernoulli tracks from Poisson
    EXPECT_GT(pmbm.num_tracks(), 0) << "Measurements should spawn new Bernoulli tracks from Poisson";
}
