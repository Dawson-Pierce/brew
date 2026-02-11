#include <gtest/gtest.h>
#include "brew/fusion/merge.hpp"

using namespace brew;

TEST(MergeGaussian, TwoClose) {
    distributions::Mixture<distributions::Gaussian> mix;
    Eigen::VectorXd m1(2), m2(2);
    m1 << 0.0, 0.0;
    m2 << 0.1, 0.1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<distributions::Gaussian>(m1, cov), 0.5);
    mix.add_component(std::make_unique<distributions::Gaussian>(m2, cov), 0.3);

    fusion::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    // Merged weight should be sum
    EXPECT_NEAR(mix.weight(0), 0.8, 1e-10);
    // Merged mean should be weighted average
    EXPECT_NEAR(mix.component(0).mean()(0), (0.5 * 0.0 + 0.3 * 0.1) / 0.8, 1e-10);
}

TEST(MergeGaussian, TwoFar) {
    distributions::Mixture<distributions::Gaussian> mix;
    Eigen::VectorXd m1(2), m2(2);
    m1 << 0.0, 0.0;
    m2 << 100.0, 100.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<distributions::Gaussian>(m1, cov), 0.5);
    mix.add_component(std::make_unique<distributions::Gaussian>(m2, cov), 0.3);

    fusion::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 2u);
}

TEST(MergeGGIW, TwoClose) {
    distributions::Mixture<distributions::GGIW> mix;
    Eigen::VectorXd m1(4), m2(4);
    m1 << 0.0, 0.0, 1.0, 0.0;
    m2 << 0.1, 0.1, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<distributions::GGIW>(m1, cov, 10.0, 5.0, 10.0, V), 0.6);
    mix.add_component(std::make_unique<distributions::GGIW>(m2, cov, 12.0, 6.0, 11.0, V), 0.4);

    fusion::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    EXPECT_NEAR(mix.weight(0), 1.0, 1e-10);
}

TEST(MergeTrajectoryGaussian, SameSize) {
    distributions::Mixture<distributions::TrajectoryGaussian> mix;
    Eigen::VectorXd m1(4), m2(4);
    m1 << 0.0, 0.0, 1.0, 0.0;
    m2 << 0.1, 0.1, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    mix.add_component(std::make_unique<distributions::TrajectoryGaussian>(0, 4, m1, cov), 0.5);
    mix.add_component(std::make_unique<distributions::TrajectoryGaussian>(0, 4, m2, cov), 0.3);

    fusion::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    EXPECT_NEAR(mix.weight(0), 0.8, 1e-10);
}

TEST(MergeTrajectoryGGIW, SameSize) {
    distributions::Mixture<distributions::TrajectoryGGIW> mix;
    Eigen::VectorXd m1(4), m2(4);
    m1 << 0.0, 0.0, 1.0, 0.0;
    m2 << 0.1, 0.1, 1.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<distributions::TrajectoryGGIW>(0, 4, m1, cov, 10.0, 5.0, 10.0, V), 0.5);
    mix.add_component(std::make_unique<distributions::TrajectoryGGIW>(0, 4, m2, cov, 12.0, 6.0, 11.0, V), 0.3);

    fusion::merge(mix, 4.0);

    EXPECT_EQ(mix.size(), 1u);
    EXPECT_NEAR(mix.weight(0), 0.8, 1e-10);
}
