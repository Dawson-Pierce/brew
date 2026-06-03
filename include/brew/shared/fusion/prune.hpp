#pragma once

#include "brew/shared/mixture.hpp"

namespace brew::fusion {

template <typename T, int N>
void prune(models::Mixture<T, N>& mixture, double threshold) {
    for (std::size_t i = mixture.size(); i > 0; --i) {
        if (mixture.weight(i - 1) < threshold) {
            mixture.remove_component(i - 1);
        }
    }
}

}
