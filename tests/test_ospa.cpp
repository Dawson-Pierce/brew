#include <gtest/gtest.h>
#include "brew/metrics/ospa.hpp"

using namespace brew::metrics;

TEST(OSPA, IdenticalSetsZeroDistance) {
    std::vector<Eigen::VectorXd> est, truth;
    Eigen::VectorXd a(2), b(2);
    a << 1.0, 2.0;
    b << 3.0, 4.0;
    est.push_back(a);
    est.push_back(b);
    truth.push_back(a);
    truth.push_back(b);

    auto result = calculate_ospa(est, truth, 10.0, 2);
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
    EXPECT_NEAR(result.localization, 0.0, 1e-10);
    EXPECT_NEAR(result.cardinality, 0.0, 1e-10);
}

TEST(OSPA, EmptyVsNonEmptyGivesCutoff) {
    std::vector<Eigen::VectorXd> est, truth;
    Eigen::VectorXd a(2);
    a << 1.0, 2.0;
    truth.push_back(a);

    auto result = calculate_ospa(est, truth, 5.0, 2);
    EXPECT_NEAR(result.distance, 5.0, 1e-10);
    EXPECT_NEAR(result.cardinality, 5.0, 1e-10);
}

TEST(OSPA, BothEmptyGivesZero) {
    std::vector<Eigen::VectorXd> est, truth;
    auto result = calculate_ospa(est, truth, 10.0, 2);
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
}

TEST(OSPA, CardinalityMismatchPenalty) {
    std::vector<Eigen::VectorXd> est, truth;
    Eigen::VectorXd a(2);
    a << 0.0, 0.0;
    est.push_back(a);
    truth.push_back(a);
    // Add extra truth target far away
    Eigen::VectorXd b(2);
    b << 100.0, 100.0;
    truth.push_back(b);

    double cutoff = 10.0;
    auto result = calculate_ospa(est, truth, cutoff, 2);
    // Should have non-zero cardinality component due to mismatch
    EXPECT_GT(result.cardinality, 0.0);
    EXPECT_GT(result.distance, 0.0);
}

TEST(GOSPA, IdenticalSetsZeroDistance) {
    std::vector<Eigen::VectorXd> est, truth;
    Eigen::VectorXd a(2);
    a << 1.0, 2.0;
    est.push_back(a);
    truth.push_back(a);

    auto result = calculate_gospa(est, truth, 10.0, 2);
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
    EXPECT_NEAR(result.localization, 0.0, 1e-10);
    EXPECT_NEAR(result.cardinality, 0.0, 1e-10);
}

TEST(GOSPA, EmptyVsNonEmpty) {
    std::vector<Eigen::VectorXd> est, truth;
    Eigen::VectorXd a(2);
    a << 1.0, 2.0;
    truth.push_back(a);

    auto result = calculate_gospa(est, truth, 5.0, 2, 2.0);
    EXPECT_GT(result.distance, 0.0);
    EXPECT_GT(result.cardinality, 0.0);
}

TEST(GOSPA, BothEmptyGivesZero) {
    std::vector<Eigen::VectorXd> est, truth;
    auto result = calculate_gospa(est, truth, 10.0, 2);
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
}
