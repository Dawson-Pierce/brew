#include <gtest/gtest.h>
#include "brew/shared/fusion/arithmetic_average.hpp"
#include "brew/gaussian/gci.hpp"
#include "brew/gaussian/gaussian_model.hpp"
#include "brew/shared/mixture.hpp"

using namespace brew;
using G = models::Gaussian<>;
using Mix = models::Mixture<G>;

static Mix one(const Eigen::Vector2d& m, const Eigen::Matrix2d& P, double w = 1.0) {
    Mix mix;
    mix.add_component(std::make_unique<G>(m, P), w);
    return mix;
}

TEST(Fusion, ArithmeticAverageUnionAndScale) {
    Mix a = one({0, 0}, Eigen::Matrix2d::Identity());
    Mix b = one({5, 5}, Eigen::Matrix2d::Identity());
    auto f = fusion::arithmetic_average<G>({&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 2u);
    EXPECT_NEAR(f.weight(0), 0.5, 1e-12);
    EXPECT_NEAR(f.weight(1), 0.5, 1e-12);
}

TEST(Fusion, GciIdenticalIsIdentity) {
    Eigen::Vector2d m(1, 2);
    Eigen::Matrix2d P;
    P << 2, 0.3, 0.3, 1;
    Mix a = one(m, P), b = one(m, P);
    auto f = gaussian::gci<>({&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).mean().isApprox(m, 1e-9));
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-9));
    EXPECT_NEAR(f.weight(0), 1.0, 1e-9);
}

TEST(Fusion, GciSameCovAveragesMean) {
    Eigen::Vector2d m1(0, 0), m2(4, 2);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 3.0;
    Mix a = one(m1, P), b = one(m2, P);
    auto f = gaussian::gci<>({&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).mean().isApprox((m1 + m2) * 0.5, 1e-9));
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-9));
}

TEST(Fusion, GciDiffCovInformationForm) {
    Eigen::Vector2d z = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P1 = Eigen::Matrix2d::Identity() * 2.0, P2;
    P2 << 1, 0, 0, 4;
    Mix a = one(z, P1), b = one(z, P2);
    const double w = 0.5;
    auto f = gaussian::gci<>({&a, &b}, {w, 1 - w});
    Eigen::Matrix2d expected_info = w * P1.inverse() + (1 - w) * P2.inverse();
    EXPECT_TRUE(f.component(0).covariance().inverse().isApprox(expected_info, 1e-9));
}

TEST(Fusion, GciMixtureCombinatorialNormalized) {
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity();
    Mix a, b;
    a.add_component(std::make_unique<G>(Eigen::Vector2d(0, 0), P), 0.5);
    a.add_component(std::make_unique<G>(Eigen::Vector2d(1, 0), P), 0.5);
    b.add_component(std::make_unique<G>(Eigen::Vector2d(0, 1), P), 0.5);
    b.add_component(std::make_unique<G>(Eigen::Vector2d(1, 1), P), 0.5);
    auto f = gaussian::gci<>({&a, &b}, {0.5, 0.5});
    EXPECT_EQ(f.size(), 4u);
    double sw = 0;
    for (std::size_t i = 0; i < f.size(); ++i) sw += f.weight(i);
    EXPECT_NEAR(sw, 1.0, 1e-9);
}
