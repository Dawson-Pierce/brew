#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include "brew/utils/assert.hpp"
#include <array>
#include <limits>

namespace brew::models {
// A positive-impulse posterior component; infinite variance denotes the uniform
// prior on [0, maximum]. Mixture weights are conditional on this object's density.
struct FiringPosteriorComponent {
    double weight = 1.0, probability = 0.0;
    double observation = 0.0, variance = std::numeric_limits<double>::infinity();
    double maximum = 0.0;
};

// Downstream marks: these do not alter pose likelihoods or PHD mass.
struct FiringEstimate {
    double impulse_probability = std::numeric_limits<double>::quiet_NaN();
    double impulse_magnitude = std::numeric_limits<double>::quiet_NaN();
    double impulse_std = std::numeric_limits<double>::quiet_NaN();
    double impulse_lower = std::numeric_limits<double>::quiet_NaN();
    double impulse_upper = std::numeric_limits<double>::quiet_NaN();
    std::vector<FiringPosteriorComponent> firing_components;

};

}

namespace brew::utils::firing_detail {
inline double log_add(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

inline double log_cdf(double x) {
    if (x > -30.0) return std::log(0.5 * std::erfc(-x / std::sqrt(2.0)));
    // Mills-ratio expansion avoids underflow in the negative tail.
    long double sum = 1, term = 1, z = x;
    for (int n = 1; n < 100; ++n) {
        long double next = -term * (2 * n - 1) / (z * z);
        if (std::abs(next) >= std::abs(term)) break;
        sum += next;
        term = next;
        if (std::abs(term) < 1e-19L * std::abs(sum)) break;
    }
    return static_cast<double>(-0.5L * z * z - std::log(-z)
        - 0.5L * std::log(2.0L * M_PI) + std::log(sum));
}

inline double log_normal_mass(double lower, double upper) {
    const double large = lower > 0 ? log_cdf(-lower) : log_cdf(upper);
    const double small = lower > 0 ? log_cdf(-upper) : log_cdf(lower);
    return large + std::log(-std::expm1(small - large));
}

inline std::array<double, 2> impulse_log_evidence(double observation, double variance, double impulse_max) {
    BREW_ASSERT(variance > 0, "Impulse likelihood variance must be positive");
    const double sigma = std::sqrt(variance);
    return {-0.5 * (observation * observation / variance + std::log(2 * M_PI * variance)),
            log_normal_mass(-observation / sigma, (impulse_max - observation) / sigma)
                - std::log(impulse_max)};
}

inline models::FiringEstimate impulse_posterior(double observation, double variance,
                                     double probability, double impulse_max) {
    models::FiringEstimate result;
    if (!std::isfinite(observation)) return result;
    BREW_ASSERT(variance > 0, "Impulse likelihood variance must be positive");
    result.impulse_probability = probability;
    const double sigma = std::sqrt(variance);
    const double a = -observation / sigma, b = (impulse_max - observation) / sigma;
    const double mass = log_normal_mass(a, b);
    const double pa = std::exp(-0.5 * a * a - 0.5 * std::log(2 * M_PI) - mass);
    const double pb = std::exp(-0.5 * b * b - 0.5 * std::log(2 * M_PI) - mass);
    const double shift = pa - pb;
    const double mean = observation + sigma * shift;
    // Match the stable centered expression used for a truncated normal.
    const double var = variance * (1 + (a - shift) * pa - (b - shift) * pb);
    result.impulse_magnitude = probability * mean;
    result.impulse_std = std::sqrt(std::max(0.0, probability * (var + mean * mean)
                                           - result.impulse_magnitude * result.impulse_magnitude));
    const auto quantile = [&](double q) {
        if (q <= 1 - probability) return 0.0;
        const double target = std::log((q - (1 - probability)) / probability);
        double lo = 0, hi = impulse_max;
        for (int i = 0; i < 80; ++i) {
            const double mid = 0.5 * (lo + hi);
            const double cdf = log_normal_mass(a, (mid - observation) / sigma) - mass;
            if (cdf < target) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    };
    result.impulse_lower = quantile(0.025);
    result.impulse_upper = quantile(0.975);
    result.firing_components = {{1.0, probability, observation, variance, impulse_max}};
    return result;
}

inline models::FiringEstimate firing_prior(double probability, double maximum) {
    models::FiringEstimate result;
    result.impulse_probability = probability;
    result.impulse_magnitude = probability * maximum / 2.0;
    result.impulse_std = std::sqrt(probability * maximum * maximum / 3.0
                                 - result.impulse_magnitude * result.impulse_magnitude);
    const auto quantile = [&](double q) {
        return q <= 1.0 - probability ? 0.0 : maximum * (q - (1.0 - probability)) / probability;
    };
    result.impulse_lower = quantile(.025); result.impulse_upper = quantile(.975);
    result.firing_components = {{1.0, probability, 0.0, INFINITY, maximum}};
    return result;
}

// Quantiles of a merged spike-and-slab distribution, not averages of intervals.
inline std::array<double, 2> firing_interval(const std::vector<models::FiringPosteriorComponent>& components) {
    if (components.empty()) return {NAN, NAN};
    double spike = 0.0, maximum = 0.0;
    for (const auto& c : components) {
        spike += c.weight * (1.0 - c.probability);
        maximum = std::max(maximum, c.maximum);
    }
    const auto quantile = [&](double q) {
        if (q <= spike) return 0.0;
        double lo = 0.0, hi = maximum;
        for (int i = 0; i < 80; ++i) {
            const double mid = .5 * (lo + hi);
            double cdf = spike;
            for (const auto& c : components) {
                double mass = 1.0;
                if (mid < c.maximum) {
                    if (std::isinf(c.variance)) mass = mid / c.maximum;
                    else {
                        const double sigma = std::sqrt(c.variance), a = -c.observation / sigma;
                        mass = std::exp(log_normal_mass(a, (mid - c.observation) / sigma)
                                      - log_normal_mass(a, (c.maximum - c.observation) / sigma));
                    }
                }
                cdf += c.weight * c.probability * std::clamp(mass, 0.0, 1.0);
            }
            if (cdf < q) lo = mid; else hi = mid;
        }
        return .5 * (lo + hi);
    };
    return {quantile(.025), quantile(.975)};
}


}

namespace brew::models {
struct FiringMerge {
    static FiringEstimate apply(const std::vector<std::pair<const FiringEstimate*, double>>& sources) {
        if (sources.size() == 1 && sources[0].first) return *sources[0].first;
        double total = 0;
        for (const auto& [state, weight] : sources) total += weight;
        FiringEstimate result;
        if (total <= 0) return result;
        bool complete = true;
        double burn = 0, mean = 0, second = 0;
        for (const auto& [state, weight] : sources) {
            if (weight == 0) continue;
            const FiringEstimate empty;
            const auto& value = state ? *state : empty;
            const double q = weight / total;
            complete = complete && !value.firing_components.empty();
            for (auto component : value.firing_components) {
                component.weight *= q;
                result.firing_components.push_back(component);
            }
            burn += q * value.impulse_probability;
            mean += q * value.impulse_magnitude;
            second += q * (value.impulse_std * value.impulse_std + value.impulse_magnitude * value.impulse_magnitude);
        }
        result.impulse_probability = burn;
        result.impulse_magnitude = mean;
        result.impulse_std = std::isfinite(second) ? std::sqrt(std::max(0.0, second - mean * mean))
                                                 : std::numeric_limits<double>::quiet_NaN();
        if (!complete) result.firing_components.clear();
        const auto interval = utils::firing_detail::firing_interval(result.firing_components);
        result.impulse_lower = interval[0]; result.impulse_upper = interval[1];
        return result;
    }
};
}
