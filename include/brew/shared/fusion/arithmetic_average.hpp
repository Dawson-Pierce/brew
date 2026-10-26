#pragma once

// Arithmetic-average (linear opinion pool) fusion: f = sum_i w_i f_i.
// Model-agnostic — for any mixture it is the union of all source components with
// each source's weights scaled by w_i. Weights default to uniform 1/M.
#include "brew/shared/mixture.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace brew::fusion {

template <typename T, int N = Eigen::Dynamic>
models::Mixture<T, N> arithmetic_average(
    const std::vector<const models::Mixture<T, N>*>& sources,
    std::vector<double> weights = {}) {
    models::Mixture<T, N> out;
    const std::size_t M = sources.size();
    if (M == 0) return out;
    if (weights.size() != M) weights.assign(M, 1.0);

    double weight_sum = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
        if (!std::isfinite(weights[i]) || weights[i] < 0.0) {
            throw std::invalid_argument("Arithmetic-average source weights must be finite and nonnegative.");
        }
        if (sources[i] && !sources[i]->empty()) weight_sum += weights[i];
    }
    if (!(weight_sum > 0.0)) return out;

    for (std::size_t i = 0; i < M; ++i) {
        if (!sources[i] || sources[i]->empty()) continue;
        const auto& mix = *sources[i];
        for (std::size_t k = 0; k < mix.size(); ++k) {
            out.add_component(mix.component(k).clone_typed(),
                              (weights[i] / weight_sum) * mix.weight(k));
        }
    }
    return out;
}

}  // namespace brew::fusion
