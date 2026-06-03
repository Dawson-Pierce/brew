#pragma once

#include "brew/shared/mixture.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

namespace brew::fusion {

template <typename T, int N>
void cap(models::Mixture<T, N>& mixture, std::size_t max_components) {
    if (mixture.size() <= max_components) return;

    std::vector<std::size_t> indices(mixture.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        return mixture.weight(a) > mixture.weight(b);
    });

    std::vector<std::size_t> to_remove(indices.begin() + static_cast<long>(max_components), indices.end());
    std::sort(to_remove.rbegin(), to_remove.rend());

    for (auto idx : to_remove) {
        mixture.remove_component(idx);
    }
}

}
