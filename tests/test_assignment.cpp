#include <gtest/gtest.h>
#include "brew/assignment/hungarian.hpp"
#include "brew/assignment/murty.hpp"
#include <cmath>
#include <limits>

using namespace brew::assignment;

TEST(Hungarian, Simple3x3) {
    // Classic 3x3 assignment problem
    Eigen::MatrixXd cost(3, 3);
    cost << 1, 2, 3,
            2, 4, 6,
            3, 6, 9;
    auto result = hungarian(cost);
    // Optimal: (0,0)=1, (1,1)=4, (2,2)=9 => 14
    // or (0,2)=3, (1,1)=4, (2,0)=3 => 10
    // or (0,1)=2, (1,0)=2, (2,2)=9 => 13
    // or (0,2)=3, (1,0)=2, (2,1)=6 => 11
    // or (0,0)=1, (1,2)=6, (2,1)=6 => 13
    // or (0,1)=2, (1,2)=6, (2,0)=3 => 11
    // Minimum is 10: (0,2), (1,1), (2,0)
    EXPECT_EQ(result.assignments.size(), 3u);
    EXPECT_NEAR(result.total_cost, 10.0, 1e-10);
}

TEST(Hungarian, IdentityCost) {
    // Diagonal costs are cheapest
    Eigen::MatrixXd cost(3, 3);
    cost << 1, 10, 10,
           10,  1, 10,
           10, 10,  1;
    auto result = hungarian(cost);
    EXPECT_NEAR(result.total_cost, 3.0, 1e-10);
}

TEST(Hungarian, Rectangular) {
    // 2 rows, 4 columns: assign 2 rows to 2 of the 4 columns
    Eigen::MatrixXd cost(2, 4);
    cost << 5, 1, 3, 2,
            2, 4, 6, 1;
    auto result = hungarian(cost);
    // Optimal: row 0->col 1 (1), row 1->col 3 (1) => total 2
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
    // Results should be in ascending cost order
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i].total_cost, results[i-1].total_cost - 1e-10);
    }
    // First should be optimal (10.0)
    EXPECT_NEAR(results[0].total_cost, 10.0, 1e-10);
}

TEST(Murty, RectangularKBest) {
    Eigen::MatrixXd cost(2, 3);
    cost << 1, 3, 5,
            4, 2, 6;
    auto results = murty(cost, 3);
    ASSERT_GE(results.size(), 2u);
    // Optimal: row 0->col 0 (1), row 1->col 1 (2) => 3
    EXPECT_NEAR(results[0].total_cost, 3.0, 1e-10);
}
