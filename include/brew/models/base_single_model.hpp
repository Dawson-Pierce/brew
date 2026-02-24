#pragma once

// Ported from: +BREW/+distributions/BaseSingleModel.m
// Original name: BaseSingleModel
// Ported on: 2026-02-07
// Notes: Pure data base class — clone() and virtual destructor only.

#include <Eigen/Dense>
#include <memory>

namespace brew::models {

/// Abstract base class for single distribution models.
/// Distributions are pure parameter holders — no sampling or pdf evaluation.
class BaseSingleModel {
public:
    virtual ~BaseSingleModel() = default;

    /// Deep copy (replaces MATLAB matlab.mixin.Copyable)
    [[nodiscard]] virtual std::unique_ptr<BaseSingleModel> clone() const = 0;

    /// Whether this distribution models an extended target (e.g. has extent parameters).
    [[nodiscard]] virtual bool is_extended() const { return false; }
};

} // namespace brew::models
