#pragma once

// Shared utilities for labeled/unlabeled multi-Bernoulli filters.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

namespace brew::multi_target::detail {

struct SubsetSolution {
    double cost;
    std::vector<int> indices;
};

inline std::vector<SubsetSolution> k_shortest_subsets(
    const Eigen::VectorXd& costs, int k)
{
    std::vector<SubsetSolution> result;
    if (k <= 0) return result;
    const int n = static_cast<int>(costs.size());
    if (n == 0) { result.push_back({0.0, {}}); return result; }

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
        [&](int a, int b) { return costs(a) < costs(b); });

    struct Node {
        double sum;
        int last;
        std::vector<int> chosen_sorted;
        bool operator>(const Node& o) const { return sum > o.sum; }
    };
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({0.0, -1, {}});

    result.reserve(static_cast<std::size_t>(k));
    while (!pq.empty() && static_cast<int>(result.size()) < k) {
        Node top = pq.top(); pq.pop();
        std::vector<int> orig;
        orig.reserve(top.chosen_sorted.size());
        for (int si : top.chosen_sorted) orig.push_back(perm[si]);
        std::sort(orig.begin(), orig.end());
        result.push_back({top.sum, std::move(orig)});

        for (int i = top.last + 1; i < n; ++i) {
            Node next;
            next.sum = top.sum + costs(perm[i]);
            next.last = i;
            next.chosen_sorted = top.chosen_sorted;
            next.chosen_sorted.push_back(i);
            pq.push(std::move(next));
        }
    }
    return result;
}

inline double log_sum_exp(const std::vector<double>& v) {
    if (v.empty()) return -std::numeric_limits<double>::infinity();
    double m = v[0];
    for (double x : v) if (x > m) m = x;
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

template <typename U, typename = void>
struct has_get_last_state : std::false_type {};

template <typename U>
struct has_get_last_state<U,
    std::void_t<decltype(std::declval<const U&>().get_last_state())>
> : std::true_type {};

template <typename U>
Eigen::VectorXd get_state_vector(const U& dist) {
    if constexpr (has_get_last_state<U>::value) {
        return dist.get_last_state();
    } else {
        return dist.mean();
    }
}

}
