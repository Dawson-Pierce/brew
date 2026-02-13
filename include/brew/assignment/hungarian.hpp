#pragma once

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace brew::assignment {

struct AssignmentResult {
    std::vector<std::pair<int, int>> assignments; // (row, col) pairs
    double total_cost = 0.0;
};

/// Hungarian (Munkres) algorithm for optimal assignment.
/// Input: cost matrix (n_rows x n_cols). Entries may be infinity for forbidden assignments.
/// Returns the minimum-cost assignment and its total cost.
/// Handles rectangular matrices by padding to square internally.
[[nodiscard]] inline AssignmentResult hungarian(const Eigen::MatrixXd& cost) {
    const int orig_rows = static_cast<int>(cost.rows());
    const int orig_cols = static_cast<int>(cost.cols());

    if (orig_rows == 0 || orig_cols == 0) {
        return {};
    }

    const int n = std::max(orig_rows, orig_cols);
    const double INF = std::numeric_limits<double>::infinity();
    const double BIG = 1e15; // large but finite cost for padding

    // Pad to square matrix
    Eigen::MatrixXd C = Eigen::MatrixXd::Constant(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < orig_rows && j < orig_cols) {
                C(i, j) = std::isinf(cost(i, j)) ? BIG : cost(i, j);
            }
            // padding entries stay 0 (dummy assignments, no real cost)
        }
    }

    // Kuhn-Munkres with potentials (u, v) and augmenting paths
    // 1-indexed internally for convenience
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0);    // p[j] = row assigned to col j
    std::vector<int> way(n + 1, 0);  // way[j] = prev col in augmenting path

    for (int i = 1; i <= n; ++i) {
        std::vector<double> minv(n + 1, INF);
        std::vector<bool> used(n + 1, false);
        p[0] = i;
        int j0 = 0;

        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = INF;
            int j1 = -1;

            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    double cur = C(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        // Augment along the path
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // Extract assignment (convert back to 0-indexed, skip padding)
    AssignmentResult result;
    result.total_cost = 0.0;

    for (int j = 1; j <= n; ++j) {
        int row = p[j] - 1;
        int col = j - 1;
        if (row < orig_rows && col < orig_cols) {
            result.assignments.emplace_back(row, col);
            result.total_cost += cost(row, col);
        }
    }

    return result;
}

} // namespace brew::assignment
