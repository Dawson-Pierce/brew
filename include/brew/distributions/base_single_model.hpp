#pragma once

// Ported from: +BREW/+distributions/BaseSingleModel.m
// Original name: BaseSingleModel
// Ported on: 2026-02-07
// Notes: Pure data base class — clone() and virtual destructor only.

#include <Eigen/Dense>
#include <memory>

namespace brew::distributions {

/// Abstract base class for single distribution models.
/// Distributions are pure parameter holders — no sampling or pdf evaluation.
class BaseSingleModel {
public:
    virtual ~BaseSingleModel() = default;

    /// Deep copy (replaces MATLAB matlab.mixin.Copyable)
    [[nodiscard]] virtual std::unique_ptr<BaseSingleModel> clone() const = 0;
};

} // namespace brew::distributions
