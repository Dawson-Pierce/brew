#pragma once

#include "brew/shared/mixture.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace brew::fusion {

template <typename T, int N>
void cap(models::Mixture<T, N>& mixture, std::size_t max_components) {
    if (mixture.size() <= max_components) return;

    std::vector<std::size_t> indices(mixture.size());
    std::iota(indices.begin(), indices.end(), 0);
    // NaN/inf-safe strict weak ordering: non-finite weights sort last (and get capped away).
    // A raw `weight(a) > weight(b)` comparator is undefined behaviour when any weight is NaN
    // (NaN compares false both ways), which corrupts std::sort and hangs.
    std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        const double wa = mixture.weight(a), wb = mixture.weight(b);
        const bool fa = std::isfinite(wa), fb = std::isfinite(wb);
        if (fa != fb) return fa;
        if (!fa) return a < b;
        return wa > wb;
    });

    std::vector<std::size_t> to_remove(indices.begin() + static_cast<long>(max_components), indices.end());
    std::sort(to_remove.rbegin(), to_remove.rend());

    for (auto idx : to_remove) {
        mixture.remove_component(idx);
    }
}

}
