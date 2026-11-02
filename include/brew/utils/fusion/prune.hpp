#pragma once

#include "brew/models/mixture.hpp"

namespace brew::utils::fusion {

template <typename T, int N>
void prune(models::Mixture<T, N>& mixture, double threshold) {
    for (std::size_t i = mixture.size(); i > 0; --i) {
        if (mixture.weight(i - 1) < threshold) {
            mixture.remove_component(i - 1);
        }
    }
}

}
