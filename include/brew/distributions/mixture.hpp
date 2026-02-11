#pragma once

// Ported from: +BREW/+distributions/BaseMixtureModel.m
// Original name: BaseMixtureModel
// Ported on: 2026-02-07
// Notes: Implemented as a generic template container.

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <vector>

namespace brew::distributions {

/// Generic mixture model container.
/// T must provide: std::unique_ptr<T> clone_typed() const;
///
/// Replaces all per-type mixture classes (GaussianMixture, GGIWMixture, etc.)
/// with a single generic container. Merge/prune/cap operations live in the
/// fusion package as free functions.
template <typename T>
class Mixture {
public:
    Mixture() = default;

    [[nodiscard]] std::unique_ptr<Mixture<T>> clone() const {
        auto result = std::make_unique<Mixture<T>>();
        result->components_.reserve(components_.size());
        for (const auto& c : components_) {
            result->components_.push_back(c->clone_typed());
        }
        result->weights_ = weights_;
        return result;
    }

    // ---- Component access ----

    [[nodiscard]] std::size_t size() const { return components_.size(); }
    [[nodiscard]] bool empty() const { return components_.empty(); }
    [[nodiscard]] std::size_t length() const { return components_.size(); }

    [[nodiscard]] T& component(std::size_t i) { return *components_[i]; }
    [[nodiscard]] const T& component(std::size_t i) const { return *components_[i]; }

    [[nodiscard]] double weight(std::size_t i) const { return weights_(static_cast<Eigen::Index>(i)); }
    [[nodiscard]] const Eigen::VectorXd& weights() const { return weights_; }
    [[nodiscard]] Eigen::VectorXd& weights() { return weights_; }

    // ---- Component management ----

    void add_component(std::unique_ptr<T> dist, double weight) {
        components_.push_back(std::move(dist));
        weights_.conservativeResize(weights_.size() + 1);
        weights_(weights_.size() - 1) = weight;
    }

    void add_components(const Mixture<T>& other) {
        for (std::size_t i = 0; i < other.size(); ++i) {
            components_.push_back(other.components_[i]->clone_typed());
            weights_.conservativeResize(weights_.size() + 1);
            weights_(weights_.size() - 1) = other.weights_(static_cast<Eigen::Index>(i));
        }
    }

    void remove_component(std::size_t idx) {
        components_.erase(components_.begin() + static_cast<long>(idx));
        Eigen::VectorXd new_weights(weights_.size() - 1);
        for (Eigen::Index j = 0, k = 0; j < weights_.size(); ++j) {
            if (static_cast<std::size_t>(j) != idx) {
                new_weights(k++) = weights_(j);
            }
        }
        weights_ = new_weights;
    }

    void remove_components(const std::vector<std::size_t>& indices) {
        if (indices.empty()) return;
        std::vector<std::size_t> sorted = indices;
        std::sort(sorted.rbegin(), sorted.rend());
        for (auto idx : sorted) {
            remove_component(idx);
        }
    }

    void clear() {
        components_.clear();
        weights_.resize(0);
    }

    [[nodiscard]] Mixture<T> extract_mix(double threshold) const {
        Mixture<T> result;
        for (std::size_t i = 0; i < size(); ++i) {
            if (weight(i) >= threshold) {
                result.add_component(components_[i]->clone_typed(), weight(i));
            }
        }
        return result;
    }

    // ---- Direct access to underlying storage ----

    [[nodiscard]] const std::vector<std::unique_ptr<T>>& components() const { return components_; }
    [[nodiscard]] std::vector<std::unique_ptr<T>>& components() { return components_; }

private:
    std::vector<std::unique_ptr<T>> components_;
    Eigen::VectorXd weights_;
};

} // namespace brew::distributions
