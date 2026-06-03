#include <gtest/gtest.h>
#include "brew/assignment/hungarian.hpp"
#include "brew/assignment/murty.hpp"
#include "brew/assignment/gibbs.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace brew::assignment;

namespace {

Eigen::MatrixXd glmb_cost(int n_rows, int num_meas) {
    Eigen::MatrixXd c(n_rows, num_meas + n_rows);
    c.setConstant(std::numeric_limits<double>::infinity());
    for (int r = 0; r < n_rows; ++r) c(r, num_meas + r) = 0.0;
    return c;
}
}

TEST(Hungarian, Simple3x3) {

    Eigen::MatrixXd cost(3, 3);
    cost << 1, 2, 3,
            2, 4, 6,
            3, 6, 9;
    auto result = hungarian(cost);

    EXPECT_EQ(result.assignments.size(), 3u);
    EXPECT_NEAR(result.total_cost, 10.0, 1e-10);
}

TEST(Hungarian, IdentityCost) {

    Eigen::MatrixXd cost(3, 3);
    cost << 1, 10, 10,
           10,  1, 10,
           10, 10,  1;
    auto result = hungarian(cost);
    EXPECT_NEAR(result.total_cost, 3.0, 1e-10);
}

TEST(Hungarian, Rectangular) {

    Eigen::MatrixXd cost(2, 4);
    cost << 5, 1, 3, 2,
            2, 4, 6, 1;
    auto result = hungarian(cost);

    EXPECT_EQ(result.assignments.size(), 2u);
    EXPECT_NEAR(result.total_cost, 2.0, 1e-10);
}

TEST(Hungarian, WithInfinity) {
    Eigen::MatrixXd cost(2, 2);
    double INF = std::numeric_limits<double>::infinity();
    cost << 1, INF,
           INF, 1;
    auto result = hungarian(cost);
    EXPECT_NEAR(result.total_cost, 2.0, 1e-10);
}

TEST(Hungarian, SingleElement) {
    Eigen::MatrixXd cost(1, 1);
    cost << 5.0;
    auto result = hungarian(cost);
    EXPECT_EQ(result.assignments.size(), 1u);
    EXPECT_NEAR(result.total_cost, 5.0, 1e-10);
}

TEST(Murty, KEqualsOne) {
    Eigen::MatrixXd cost(3, 3);
    cost << 1, 2, 3,
            2, 4, 6,
            3, 6, 9;
    auto results = murty(cost, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_NEAR(results[0].total_cost, 10.0, 1e-10);
}

TEST(Murty, KBestThree) {
    Eigen::MatrixXd cost(3, 3);
    cost << 1, 2, 3,
            2, 4, 6,
            3, 6, 9;
    auto results = murty(cost, 3);
    ASSERT_GE(results.size(), 2u);

    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i].total_cost, results[i-1].total_cost - 1e-10);
    }

    EXPECT_NEAR(results[0].total_cost, 10.0, 1e-10);
}

TEST(Murty, RectangularKBest) {
    Eigen::MatrixXd cost(2, 3);
    cost << 1, 3, 5,
            4, 2, 6;
    auto results = murty(cost, 3);
    ASSERT_GE(results.size(), 2u);

    EXPECT_NEAR(results[0].total_cost, 3.0, 1e-10);
}

TEST(Gibbs, FindsBestAssignmentLikeMurty) {

    Eigen::MatrixXd cost = glmb_cost(2, 2);
    cost(0, 0) = -2.0; cost(0, 1) = 0.5;
    cost(1, 0) = 0.5;  cost(1, 1) = -3.0;

    auto g = gibbs(cost, 2, 50, 7);
    ASSERT_FALSE(g.empty());

    for (std::size_t i = 1; i < g.size(); ++i)
        EXPECT_GE(g[i].total_cost, g[i - 1].total_cost - 1e-9);
    auto m = murty(cost, 50);
    ASSERT_FALSE(m.empty());
    EXPECT_NEAR(g.front().total_cost, m.front().total_cost, 1e-9);
    EXPECT_NEAR(g.front().total_cost, -5.0, 1e-9);

    EXPECT_EQ(g.front().assignments.size(), 2u);
}

TEST(Gibbs, RespectsMeasurementExclusivity) {

    Eigen::MatrixXd cost = glmb_cost(3, 2);
    for (int r = 0; r < 3; ++r) { cost(r, 0) = -1.0; cost(r, 1) = -1.0; }

    auto g = gibbs(cost, 2, 20, 123);
    ASSERT_FALSE(g.empty());
    for (const auto& res : g) {
        std::vector<int> meas_used;
        for (auto [row, col] : res.assignments)
            if (col < 2) meas_used.push_back(col);
        std::sort(meas_used.begin(), meas_used.end());
        EXPECT_EQ(std::unique(meas_used.begin(), meas_used.end()), meas_used.end())
            << "a measurement was assigned to more than one row";
    }
}

TEST(Gibbs, AllMissedWhenMeasurementsUnfavorable) {

    Eigen::MatrixXd cost = glmb_cost(2, 2);
    cost(0, 0) = 5.0; cost(0, 1) = 5.0;
    cost(1, 0) = 5.0; cost(1, 1) = 5.0;

    auto g = gibbs(cost, 2, 16, 99);
    ASSERT_FALSE(g.empty());
    EXPECT_NEAR(g.front().total_cost, 0.0, 1e-9);
}

TEST(Gibbs, HandlesNonCanonicalMatrixWithoutCrashing) {

    Eigen::MatrixXd cost(2, 1);
    cost << 1.0,
            1.0;
    auto g = gibbs(cost, 1, 8, 3);
    for (const auto& res : g) {
        EXPECT_TRUE(std::isfinite(res.total_cost));
        for (auto [row, col] : res.assignments) {
            EXPECT_GE(col, 0);
            EXPECT_LT(col, cost.cols());
        }
    }
}
