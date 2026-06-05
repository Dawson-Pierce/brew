#pragma once

// Arithmetic-average (linear opinion pool) fusion: f = sum_i w_i f_i.
// Model-agnostic — for any mixture it is the union of all source components with
// each source's weights scaled by w_i. Weights default to uniform 1/M.
#include "brew/shared/mixture.hpp"

#include <Eigen/Dense>
#include <vector>

namespace brew::fusion {

template <typename T, int N = Eigen::Dynamic>
models::Mixture<T, N> arithmetic_average(
    const std::vector<const models::Mixture<T, N>*>& sources,
    std::vector<double> weights = {}) {
    models::Mixture<T, N> out;
    const std::size_t M = sources.size();
    if (M == 0) return out;
    if (weights.size() != M) weights.assign(M, 1.0 / static_cast<double>(M));

    for (std::size_t i = 0; i < M; ++i) {
        if (!sources[i]) continue;
        const auto& mix = *sources[i];
        for (std::size_t k = 0; k < mix.size(); ++k) {
            out.add_component(mix.component(k).clone_typed(), weights[i] * mix.weight(k));
        }
    }
    return out;
}

}  // namespace brew::fusion
