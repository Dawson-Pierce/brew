#pragma once

#include "brew/assignment/hungarian.hpp"
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdint>

namespace brew::assignment {

[[nodiscard]] inline std::vector<AssignmentResult> gibbs(
    const Eigen::MatrixXd& cost, int num_meas, int num_samples,
    std::uint64_t seed = 0x9E3779B97F4A7C15ull)
{
    const int n_rows = static_cast<int>(cost.rows());
    const int n_cols = static_cast<int>(cost.cols());
    if (n_rows == 0 || num_samples <= 0 || n_cols == 0) return {};
    if (num_meas < 0) num_meas = 0;
    if (num_meas > n_cols) num_meas = n_cols;

    std::mt19937_64 rng(seed);

    std::vector<int> gamma(n_rows, -1);
    std::vector<int> owner(static_cast<std::size_t>(num_meas), -1);

    auto missed_col = [&](int r) { return num_meas + r; };
    auto cost_of = [&](int r, int g) {
        return g < 0 ? cost(r, missed_col(r)) : cost(r, g);
    };

    std::map<std::string, AssignmentResult> uniq;
    std::vector<int> cbuf;
    std::vector<double> wbuf;

    const std::int64_t max_sweeps =
        std::max<std::int64_t>(static_cast<std::int64_t>(num_samples) * 4, 20);
    for (std::int64_t sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int r = 0; r < n_rows; ++r) {
            if (gamma[r] >= 0) owner[gamma[r]] = -1;

            cbuf.clear();
            wbuf.clear();
            double cmin = std::numeric_limits<double>::infinity();

            const bool missed_ok = (missed_col(r) < n_cols) &&
                                    std::isfinite(cost(r, missed_col(r)));
            if (missed_ok) { cbuf.push_back(-1); cmin = std::min(cmin, cost(r, missed_col(r))); }

            for (int j = 0; j < num_meas; ++j) {
                if (owner[j] == -1 && std::isfinite(cost(r, j))) {
                    cbuf.push_back(j);
                    cmin = std::min(cmin, cost(r, j));
                }
            }

            if (cbuf.empty()) { gamma[r] = -1; continue; }

            double wsum = 0.0;
            for (int g : cbuf) { double w = std::exp(-(cost_of(r, g) - cmin)); wbuf.push_back(w); wsum += w; }
            std::uniform_real_distribution<double> U(0.0, wsum);
            double u = U(rng), acc = 0.0;
            int pick = cbuf.back();
            for (std::size_t t = 0; t < cbuf.size(); ++t) { acc += wbuf[t]; if (u <= acc) { pick = cbuf[t]; break; } }

            gamma[r] = pick;
            if (pick >= 0) owner[pick] = r;
        }

        std::string key;
        key.reserve(static_cast<std::size_t>(n_rows) * 4);
        for (int r = 0; r < n_rows; ++r) { key += std::to_string(gamma[r]); key += ','; }
        if (uniq.find(key) == uniq.end()) {
            AssignmentResult res;
            res.total_cost = 0.0;
            res.assignments.reserve(static_cast<std::size_t>(n_rows));
            bool feasible = true;
            for (int r = 0; r < n_rows; ++r) {
                int col = gamma[r];
                if (col < 0) {
                    const int mc = missed_col(r);
                    if (mc >= n_cols) { feasible = false; break; }
                    col = mc;
                }
                res.assignments.emplace_back(r, col);
                res.total_cost += cost(r, col);
            }
            if (feasible && std::isfinite(res.total_cost)) {
                uniq.emplace(std::move(key), std::move(res));
            }
        }
        if (static_cast<int>(uniq.size()) >= num_samples) break;
    }

    std::vector<AssignmentResult> out;
    out.reserve(uniq.size());
    for (auto& [k, v] : uniq) out.push_back(std::move(v));
    std::sort(out.begin(), out.end(),
              [](const AssignmentResult& a, const AssignmentResult& b) {
                  return a.total_cost < b.total_cost;
              });
    if (static_cast<int>(out.size()) > num_samples) out.resize(static_cast<std::size_t>(num_samples));
    return out;
}

}
