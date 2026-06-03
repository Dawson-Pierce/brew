#include <gtest/gtest.h>
#include "brew/gaussian/merge.hpp"
#include "brew/ggiw/merge.hpp"
#include "brew/trajectory_gaussian/merge.hpp"
#include "brew/trajectory_ggiw/merge.hpp"

using namespace brew;

TEST(MergeGaussian, TwoClose) {
    models::Mixture<models::Gaussian<>> mix;
    Eigen::VectorXd m1(2), m2(2);
    m1 << 0.0, 0.0;
    m2 << 0.1, 0.1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<models::Gaussian<>>(m1, cov), 0.5);
    mix.add_component(std::make_unique<models::Gaussian<>>(m2, cov), 0.3);

    gaussian::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);

    EXPECT_NEAR(mix.weight(0), 0.8, 1e-10);

    EXPECT_NEAR(mix.component(0).mean()(0), (0.5 * 0.0 + 0.3 * 0.1) / 0.8, 1e-10);
}

TEST(MergeGaussian, TwoFar) {
    models::Mixture<models::Gaussian<>> mix;
    Eigen::VectorXd m1(2), m2(2);
    m1 << 0.0, 0.0;
    m2 << 100.0, 100.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<models::Gaussian<>>(m1, cov), 0.5);
    mix.add_component(std::make_unique<models::Gaussian<>>(m2, cov), 0.3);

    gaussian::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 2u);
}

TEST(MergeGGIW, TwoClose) {
    models::Mixture<models::GGIW<>> mix;
    Eigen::VectorXd m1(4), m2(4);
    m1 << 0.0, 0.0, 1.0, 0.0;
    m2 << 0.1, 0.1, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<models::GGIW<>>(10.0, 5.0, m1, cov, 10.0, V), 0.6);
    mix.add_component(std::make_unique<models::GGIW<>>(12.0, 6.0, m2, cov, 11.0, V), 0.4);

    ggiw::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    EXPECT_NEAR(mix.weight(0), 1.0, 1e-10);
}

TEST(MergeTrajectoryGaussian, SameSize) {
    constexpr int W = 5;
    models::Mixture<models::TrajectoryGaussian<>> mix;
    Eigen::VectorXd m1(4), m2(4);
    m1 << 0.0, 0.0, 1.0, 0.0;
    m2 << 0.1, 0.1, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    mix.add_component(std::make_unique<models::TrajectoryGaussian<>>(4, models::Gaussian<>(m1, cov), W), 0.5);
    mix.add_component(std::make_unique<models::TrajectoryGaussian<>>(4, models::Gaussian<>(m2, cov), W), 0.3);

    trajectory_gaussian::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    EXPECT_NEAR(mix.weight(0), 0.8, 1e-10);
}

TEST(MergeTrajectoryGGIW, SameSize) {
    constexpr int W = 5;
    models::Mixture<models::TrajectoryGGIW<>> mix;
    Eigen::VectorXd m1(4), m2(4);
    m1 << 0.0, 0.0, 1.0, 0.0;
    m2 << 0.1, 0.1, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<models::TrajectoryGGIW<>>(4, models::GGIW<>(10.0, 5.0, m1, cov, 10.0, V), W), 0.5);
    mix.add_component(std::make_unique<models::TrajectoryGGIW<>>(4, models::GGIW<>(12.0, 6.0, m2, cov, 11.0, V), W), 0.3);

    trajectory_ggiw::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    EXPECT_NEAR(mix.weight(0), 0.8, 1e-10);
}
