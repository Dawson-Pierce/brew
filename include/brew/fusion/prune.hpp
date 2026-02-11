#pragma once

#include "brew/distributions/mixture.hpp"

namespace brew::fusion {

/// Remove components whose weight is below the threshold.
/// Generic: works on any Mixture<T>.
template <typename T>
void prune(distributions::Mixture<T>& mixture, double threshold) {
    for (std::size_t i = mixture.size(); i > 0; --i) {
        if (mixture.weight(i - 1) < threshold) {
            mixture.remove_component(i - 1);
        }
    }
}

} // namespace brew::fusion
