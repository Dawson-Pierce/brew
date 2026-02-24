#pragma once

#include "brew/models/mixture.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

namespace brew::fusion {

/// Cap the number of components by keeping the highest-weighted ones.
/// Generic: works on any Mixture<T>.
template <typename T>
void cap(models::Mixture<T>& mixture, std::size_t max_components) {
    if (mixture.size() <= max_components) return;

    // Sort indices by weight (descending)
    std::vector<std::size_t> indices(mixture.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        return mixture.weight(a) > mixture.weight(b);
    });

    // Collect indices to remove (those beyond max_components)
    std::vector<std::size_t> to_remove(indices.begin() + static_cast<long>(max_components), indices.end());
    std::sort(to_remove.rbegin(), to_remove.rend()); // descending for safe removal

    for (auto idx : to_remove) {
        mixture.remove_component(idx);
    }
}

} // namespace brew::fusion
