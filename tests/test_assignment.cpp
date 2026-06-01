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
// Build a GLMB-style association cost matrix: n_rows x (num_meas + n_rows).
// Columns [0,num_meas) are measurements; column num_meas+r is row r's
// exclusive missed/no-target slot.
Eigen::MatrixXd glmb_cost(int n_rows, int num_meas) {
    Eigen::MatrixXd c(n_rows, num_meas + n_rows);
    c.setConstant(std::numeric_limits<double>::infinity());
    for (int r = 0; r < n_rows; ++r) c(r, num_meas + r) = 0.0;  // missed slots free
    return c;
}
}  // namespace

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

// ---- Gibbs sampler (GLMB-style association) ----

TEST(Gibbs, FindsBestAssignmentLikeMurty) {
    // 2 tracks, 2 measurements. Lower (more negative) cost == higher weight.
    Eigen::MatrixXd cost = glmb_cost(2, 2);
    cost(0, 0) = -2.0; cost(0, 1) = 0.5;
    cost(1, 0) = 0.5;  cost(1, 1) = -3.0;
    // Best joint assignment: t0->m0 (-2), t1->m1 (-3) => total -5.

    auto g = gibbs(cost, /*num_meas=*/2, /*num_samples=*/50, /*seed=*/7);
    ASSERT_FALSE(g.empty());

    // Returned ascending by cost; best matches Murty's optimum.
    for (std::size_t i = 1; i < g.size(); ++i)
        EXPECT_GE(g[i].total_cost, g[i - 1].total_cost - 1e-9);
    auto m = murty(cost, 50);
    ASSERT_FALSE(m.empty());
    EXPECT_NEAR(g.front().total_cost, m.front().total_cost, 1e-9);
    EXPECT_NEAR(g.front().total_cost, -5.0, 1e-9);

    // Every row is assigned exactly once (to a measurement or its missed slot).
    EXPECT_EQ(g.front().assignments.size(), 2u);
}

TEST(Gibbs, RespectsMeasurementExclusivity) {
    // Three tracks all favor both measurements; a measurement must never be
    // shared by two rows in any returned assignment.
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
    // Measurements far costlier than the (free) missed slots => best is all-missed.
    Eigen::MatrixXd cost = glmb_cost(2, 2);
    cost(0, 0) = 5.0; cost(0, 1) = 5.0;
    cost(1, 0) = 5.0; cost(1, 1) = 5.0;

    auto g = gibbs(cost, 2, 16, 99);
    ASSERT_FALSE(g.empty());
    EXPECT_NEAR(g.front().total_cost, 0.0, 1e-9);
}

TEST(Gibbs, HandlesNonCanonicalMatrixWithoutCrashing) {
    // A square matrix with num_meas == n_cols has NO per-row missed columns, so
    // two rows competing for one column leaves one row unassignable. This must
    // not read out of bounds (regression for the missed-column OOB); any returned
    // assignment must be feasible (in-range, finite cost).
    Eigen::MatrixXd cost(2, 1);
    cost << 1.0,
            1.0;
    auto g = gibbs(cost, /*num_meas=*/1, /*num_samples=*/8, /*seed=*/3);  // must not crash
    for (const auto& res : g) {
        EXPECT_TRUE(std::isfinite(res.total_cost));
        for (auto [row, col] : res.assignments) {
            EXPECT_GE(col, 0);
            EXPECT_LT(col, cost.cols());
        }
    }
}
