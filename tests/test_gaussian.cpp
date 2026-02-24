#include <gtest/gtest.h>
#include "brew/models/gaussian.hpp"
#include "brew/models/mixture.hpp"

using namespace brew::models;

TEST(Gaussian, ConstructAndClone) {
    Eigen::VectorXd mean(2);
    mean << 1.0, 2.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    Gaussian g(mean, cov);
    EXPECT_EQ(g.mean().size(), 2);
    EXPECT_DOUBLE_EQ(g.mean()(0), 1.0);

    auto cloned = g.clone();
    auto* gc = dynamic_cast<Gaussian*>(cloned.get());
    ASSERT_NE(gc, nullptr);
    EXPECT_DOUBLE_EQ(gc->mean()(0), 1.0);
}

TEST(GaussianMixture, AddAndSize) {
    Mixture<Gaussian> mix;
    Eigen::VectorXd mean(2);
    mean << 1.0, 2.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);

    mix.add_component(std::make_unique<Gaussian>(mean, cov), 0.5);
    mix.add_component(std::make_unique<Gaussian>(mean * 2, cov), 0.5);

    EXPECT_EQ(mix.size(), 2u);
    EXPECT_DOUBLE_EQ(mix.weight(0), 0.5);
}
