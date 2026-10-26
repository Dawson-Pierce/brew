#pragma once

#include "brew/shared/mixture.hpp"

#include <cmath>
#include <memory>
#include <utility>

namespace brew::multi_target::detail {

template <typename Component>
decltype(auto) gamma_poisson_state(Component& component) {
    if constexpr (requires { component.current(); }) {
        return (component.current());
    } else {
        return (component);
    }
}

template <typename Component>
decltype(auto) gamma_poisson_state(const Component& component) {
    if constexpr (requires { component.current(); }) {
        return (component.current());
    } else {
        return (component);
    }
}

template <typename T, int N>
std::pair<std::unique_ptr<models::Mixture<T, N>>, double>
gamma_poisson_missed_detection(
    const models::Mixture<T, N>& source,
    double probability_detection)
{
    auto result = std::make_unique<models::Mixture<T, N>>();
    double likelihood = 0.0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const auto& component = source.component(i);
        const double ordinary = source.weight(i) * (1.0 - probability_detection);
        if (ordinary > 0.0) {
            result->add_component(component.clone_typed(), ordinary);
            likelihood += ordinary;
        }

        const auto& rate = gamma_poisson_state(component);
        const double q0 = std::pow(rate.beta() / (rate.beta() + 1.0), rate.alpha());
        const double zero_count = source.weight(i) * probability_detection * q0;
        if (zero_count > 0.0) {
            auto updated = component.clone_typed();
            gamma_poisson_state(*updated).beta() += 1.0;
            result->add_component(std::move(updated), zero_count);
            likelihood += zero_count;
        }
    }
    if (likelihood > 0.0) result->weights() /= likelihood;
    return {std::move(result), likelihood};
}

}
