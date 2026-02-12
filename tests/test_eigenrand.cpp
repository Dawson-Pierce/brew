#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

TEST(EigenRandTest, GeneratesNormalMatrix) {
    Eigen::Rand::Vmt19937_64 rng(42);
    Eigen::MatrixXd samples = Eigen::Rand::normal<Eigen::MatrixXd>(500, 1, rng, 0.0, 1.0);

    const double mean = samples.mean();
    EXPECT_NEAR(mean, 0.0, 0.2);
}
