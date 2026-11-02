#include <gtest/gtest.h>
#include "brew/models/gaussian.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include "brew/multiObjectTrackers/mbm_base.hpp"
#include "brew/multiObjectTrackers/cphd.hpp"
#include "brew/multiObjectTrackers/glmb.hpp"
#include "brew/multiObjectTrackers/jglmb.hpp"
#include "brew/multiObjectTrackers/mbm.hpp"
#include "brew/multiObjectTrackers/pmbm.hpp"
#include "brew/models/merge/gaussian.hpp"
#include "brew/utils/fusion/gaussian_gci.hpp"
#include "brew/utils/fusion/arithmetic_average.hpp"
#include "brew/utils/fusion/prune.hpp"
#include "brew/utils/fusion/cap.hpp"
#include "brew/utils/clustering/dbscan.hpp"
#include "brew/dynamics/single_integrator.hpp"

using namespace brew;

TEST(PHD, GaussianPredictCorrectCleanup) {

    auto ekf = std::make_unique<filters::EKF<>>();
    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    {
        Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
        ekf->set_process_noise(G * (0.1 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose());
    }
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<brew::models::Mixture<brew::models::Gaussian<>>>();
    Eigen::VectorXd birth_mean(4);
    birth_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd birth_cov = 10.0 * Eigen::MatrixXd::Identity(4, 4);
    birth->add_component(std::make_unique<brew::models::Gaussian<>>(birth_mean, birth_cov), 0.1);

    auto intensity = birth->clone();

    brew::multiObjectTrackers::PHD<brew::models::Gaussian<>, Eigen::Dynamic> phd;
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

    phd.predict(0, 1.0);
    EXPECT_GE(phd.intensity().size(), 2u);

    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    phd.correct(meas);
    EXPECT_GE(phd.intensity().size(), 1u);

    phd.cleanup();
    EXPECT_LE(phd.intensity().size(), 20u);
    EXPECT_GE(phd.extracted_mixtures().size(), 1u);
}

TEST(PHD, Clone) {
    auto ekf = std::make_unique<filters::EKF<>>();
    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
    ekf->set_dynamics(dyn);
    {
        Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
        ekf->set_process_noise(G * Eigen::MatrixXd::Identity(G.cols(), G.cols()) * G.transpose());
    }
    ekf->set_measurement_noise(Eigen::MatrixXd::Identity(2, 2));
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);

    auto birth = std::make_unique<brew::models::Mixture<brew::models::Gaussian<>>>();
    Eigen::VectorXd m(4);
    m.setZero();
    birth->add_component(std::make_unique<brew::models::Gaussian<>>(m, Eigen::MatrixXd::Identity(4, 4)), 0.1);

    brew::multiObjectTrackers::PHD<brew::models::Gaussian<>, Eigen::Dynamic> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<brew::models::Mixture<brew::models::Gaussian<>>>());

    auto cloned = phd.clone();
    ASSERT_NE(cloned, nullptr);
}

TEST(GaussianPHDConcrete, PredictCorrectCleanup) {

    auto ekf = std::make_unique<filters::EKF<>>();
    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
    ekf->set_dynamics(dyn);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    {
        Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
        ekf->set_process_noise(G * (0.1 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose());
    }
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<brew::models::Mixture<brew::models::Gaussian<>>>();
    Eigen::VectorXd birth_mean(4);
    birth_mean << 0.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<brew::models::Gaussian<>>(birth_mean, 10.0 * Eigen::MatrixXd::Identity(4, 4)), 0.1);
    auto intensity = birth->clone();

    brew::multiObjectTrackers::PHD<brew::models::Gaussian<>, Eigen::Dynamic> phd;
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

    phd.predict(0, 1.0);
    EXPECT_GE(phd.intensity().size(), 2u);
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    phd.correct(meas);
    EXPECT_GE(phd.intensity().size(), 1u);
    phd.cleanup();
    EXPECT_LE(phd.intensity().size(), 20u);
    EXPECT_GE(phd.extracted_mixtures().size(), 1u);

    auto cloned = phd.clone();
    ASSERT_NE(cloned, nullptr);
}
