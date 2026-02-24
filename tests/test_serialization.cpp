#include <gtest/gtest.h>
#include "brew/serialization/rfs_json.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;
using json = nlohmann::json;

TEST(Serialization, GaussianRoundTrip) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd cov = 2.0 * Eigen::MatrixXd::Identity(4, 4);

    models::Gaussian g(mean, cov);
    auto j = serialization::to_json(g);
    auto g2 = serialization::gaussian_from_json(j);

    EXPECT_TRUE(g2.mean().isApprox(mean, 1e-12));
    EXPECT_TRUE(g2.covariance().isApprox(cov, 1e-12));
}

TEST(Serialization, GGIWRoundTrip) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 5.0 * Eigen::MatrixXd::Identity(2, 2);

    models::GGIW g(mean, cov, 10.0, 1.0, 7.0, V);
    auto j = serialization::to_json(g);
    auto g2 = serialization::ggiw_from_json(j);

    EXPECT_TRUE(g2.mean().isApprox(mean, 1e-12));
    EXPECT_TRUE(g2.covariance().isApprox(cov, 1e-12));
    EXPECT_NEAR(g2.alpha(), 10.0, 1e-12);
    EXPECT_NEAR(g2.beta(), 1.0, 1e-12);
    EXPECT_NEAR(g2.v(), 7.0, 1e-12);
    EXPECT_TRUE(g2.V().isApprox(V, 1e-12));
}

TEST(Serialization, MixtureGaussianRoundTrip) {
    models::Mixture<models::Gaussian> mix;
    Eigen::VectorXd m1(2), m2(2);
    m1 << 1.0, 2.0;
    m2 << 3.0, 4.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<models::Gaussian>(m1, cov), 0.6);
    mix.add_component(std::make_unique<models::Gaussian>(m2, 2.0 * cov), 0.4);

    auto j = serialization::mixture_to_json(mix);
    auto mix2 = serialization::mixture_from_json<models::Gaussian>(j);

    ASSERT_EQ(mix2->size(), 2u);
    EXPECT_NEAR(mix2->weight(0), 0.6, 1e-12);
    EXPECT_NEAR(mix2->weight(1), 0.4, 1e-12);
    EXPECT_TRUE(mix2->component(0).mean().isApprox(m1, 1e-12));
    EXPECT_TRUE(mix2->component(1).mean().isApprox(m2, 1e-12));
}

TEST(Serialization, BernoulliGaussianRoundTrip) {
    Eigen::VectorXd mean(2);
    mean << 5.0, 6.0;
    Eigen::MatrixXd cov = 3.0 * Eigen::MatrixXd::Identity(2, 2);

    models::Bernoulli<models::Gaussian> b(
        0.8, std::make_unique<models::Gaussian>(mean, cov), 42);

    auto j = serialization::bernoulli_to_json(b);
    auto b2 = serialization::bernoulli_from_json<models::Gaussian>(j);

    EXPECT_NEAR(b2->existence_probability(), 0.8, 1e-12);
    EXPECT_EQ(b2->id(), 42);
    ASSERT_TRUE(b2->has_distribution());
    EXPECT_TRUE(b2->distribution().mean().isApprox(mean, 1e-12));
}

TEST(Serialization, GLMBGaussianSerialize) {
    // Set up a minimal GLMB and run one predict/correct/cleanup cycle
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.1 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::VectorXd bm(4);
    bm.setZero();
    birth->add_component(
        std::make_unique<models::Gaussian>(bm, 10.0 * Eigen::MatrixXd::Identity(4, 4)), 0.1);

    multi_target::GLMB<models::Gaussian> glmb;
    glmb.set_filter(std::move(ekf));
    glmb.set_birth_model(std::move(birth));
    glmb.set_prob_detection(0.9);
    glmb.set_prob_survive(0.99);
    glmb.set_clutter_rate(1.0);
    glmb.set_clutter_density(1e-4);

    glmb.predict(0, 1.0);
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    glmb.correct(meas);
    glmb.cleanup();

    auto j = serialization::glmb_to_json(glmb);

    // Verify JSON structure
    EXPECT_EQ(j["filter_type"].get<std::string>(), "GLMB");
    EXPECT_NEAR(j["prob_detection"].get<double>(), 0.9, 1e-12);
    EXPECT_TRUE(j.contains("cardinality_pmf"));
    EXPECT_TRUE(j.contains("global_hypotheses"));
    EXPECT_TRUE(j.contains("extracted_mixtures"));
}

TEST(Serialization, PMBMGaussianSerialize) {
    auto ekf = std::make_unique<filters::EKF>();
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    ekf->set_dynamics(dyn);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.1 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(1.0 * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::VectorXd bm(4);
    bm.setZero();
    birth->add_component(
        std::make_unique<models::Gaussian>(bm, 10.0 * Eigen::MatrixXd::Identity(4, 4)), 0.1);

    auto poisson = std::make_unique<models::Mixture<models::Gaussian>>();

    multi_target::PMBM<models::Gaussian> pmbm;
    pmbm.set_filter(std::move(ekf));
    pmbm.set_birth_model(std::move(birth));
    pmbm.set_poisson_intensity(std::move(poisson));
    pmbm.set_prob_detection(0.9);
    pmbm.set_prob_survive(0.99);
    pmbm.set_clutter_rate(1.0);
    pmbm.set_clutter_density(1e-4);

    pmbm.predict(0, 1.0);
    Eigen::MatrixXd meas(2, 1);
    meas << 0.5, 0.3;
    pmbm.correct(meas);
    pmbm.cleanup();

    auto j = serialization::pmbm_to_json(pmbm);

    EXPECT_EQ(j["filter_type"].get<std::string>(), "PMBM");
    EXPECT_TRUE(j.contains("poisson_intensity"));
    EXPECT_TRUE(j.contains("cardinality_pmf"));
    EXPECT_TRUE(j.contains("global_hypotheses"));
}

TEST(Serialization, EigenVectorRoundTrip) {
    Eigen::VectorXd v(5);
    v << 1.1, 2.2, 3.3, 4.4, 5.5;
    auto j = serialization::vector_to_json(v);
    auto v2 = serialization::vector_from_json(j);
    EXPECT_TRUE(v2.isApprox(v, 1e-12));
}

TEST(Serialization, EigenMatrixRoundTrip) {
    Eigen::MatrixXd m(3, 2);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    auto j = serialization::matrix_to_json(m);
    auto m2 = serialization::matrix_from_json(j);
    EXPECT_TRUE(m2.isApprox(m, 1e-12));
}
