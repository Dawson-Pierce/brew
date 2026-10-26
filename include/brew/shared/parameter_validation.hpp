#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace brew::detail {

inline double positive_finite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be positive and finite.");
    }
    return value;
}

inline double unit_interval(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must lie in [0, 1].");
    }
    return value;
}

}  // namespace brew::detail
