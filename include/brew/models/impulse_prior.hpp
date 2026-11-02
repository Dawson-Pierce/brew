#pragma once

#include <cmath>
#include <stdexcept>

namespace brew::models {

// One immutable setup value for the physical Bernoulli/uniform impulse prior.
// Hypotheses may override P(fire | intent), but share the magnitude support.
struct ImpulsePrior {
    const double maximum;
    const double probability;

    ImpulsePrior(double maximum, double probability)
        : maximum(maximum), probability(probability) {
        if (!std::isfinite(maximum) || maximum <= 0 || !std::isfinite(probability)
                || probability <= 0 || probability >= 1)
            throw std::invalid_argument("Impulse prior needs a positive maximum and probability strictly between zero and one");
    }
};

} // namespace brew::models
