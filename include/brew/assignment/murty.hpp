#pragma once

#include "brew/assignment/hungarian.hpp"
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <limits>
#include <functional>

namespace brew::assignment {

/// Murty's algorithm for finding the K-best assignments.
/// Returns up to k assignments in ascending cost order.
/// Uses the Hungarian algorithm as the inner solver.
[[nodiscard]] inline std::vector<AssignmentResult> murty(
    const Eigen::MatrixXd& cost_matrix, int k)
{
    const int n_rows = static_cast<int>(cost_matrix.rows());
    const int n_cols = static_cast<int>(cost_matrix.cols());
    const double INF = std::numeric_limits<double>::infinity();

    if (n_rows == 0 || n_cols == 0 || k <= 0) {
        return {};
    }

    // A subproblem: cost matrix with some forced/excluded assignments
    struct Subproblem {
        Eigen::MatrixXd cost;
        AssignmentResult solution;

        bool operator>(const Subproblem& other) const {
            return solution.total_cost > other.solution.total_cost;
        }
    };

    // Priority queue: min-heap by total cost
    auto cmp = [](const Subproblem& a, const Subproblem& b) {
        return a.solution.total_cost > b.solution.total_cost;
    };
    std::priority_queue<Subproblem, std::vector<Subproblem>, decltype(cmp)> pq(cmp);

    // Solve the initial (unconstrained) problem
    auto initial = hungarian(cost_matrix);
    if (initial.assignments.empty()) {
        return {};
    }

    Subproblem initial_sub;
    initial_sub.cost = cost_matrix;
    initial_sub.solution = initial;
    pq.push(std::move(initial_sub));

    std::vector<AssignmentResult> results;

    while (!pq.empty() && static_cast<int>(results.size()) < k) {
        auto best = pq.top();
        pq.pop();

        results.push_back(best.solution);

        if (static_cast<int>(results.size()) >= k) break;

        // Generate subproblems by partitioning (Murty's method)
        const auto& assignments = best.solution.assignments;
        Eigen::MatrixXd constrained_cost = best.cost;

        for (std::size_t t = 0; t < assignments.size(); ++t) {
            auto [fix_row, fix_col] = assignments[t];

            // Exclude this specific assignment: set cost(fix_row, fix_col) = INF
            Eigen::MatrixXd sub_cost = constrained_cost;
            sub_cost(fix_row, fix_col) = INF;

            // Solve the subproblem
            auto sub_result = hungarian(sub_cost);

            if (!sub_result.assignments.empty() &&
                std::isfinite(sub_result.total_cost))
            {
                Subproblem sub;
                sub.cost = sub_cost;
                sub.solution = sub_result;
                pq.push(std::move(sub));
            }

            // Fix this assignment for subsequent subproblems:
            // Row fix_row must be assigned to col fix_col.
            // Exclude fix_row from all other columns and fix_col from all other rows.
            for (int j = 0; j < n_cols; ++j) {
                if (j != fix_col) constrained_cost(fix_row, j) = INF;
            }
            for (int i = 0; i < n_rows; ++i) {
                if (i != fix_row) constrained_cost(i, fix_col) = INF;
            }
        }
    }

    return results;
}

} // namespace brew::assignment
