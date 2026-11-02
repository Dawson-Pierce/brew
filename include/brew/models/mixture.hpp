#pragma once

// Notes: Implemented as a generic template container.
//
// One class handles GaussianMixture / GGIWMixture / Trajectory-mixtures / etc.,
// with dispatch on (T, capacity N).
//   - N = Eigen::Dynamic: heap-backed (std::vector<std::unique_ptr<T>> + VectorXd).
//   - N fixed: stack-backed (std::array<T, N> + Matrix<double, N, 1>) with live size_ counter.
// Reduction is exposed here; density-specific merging is supplied by MixtureMerge<T>.

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <limits>
#include <type_traits>
#include <vector>
#include "brew/models/base_single_model.hpp"

namespace brew::models { template<class T, int N> class Mixture; }
namespace brew::utils::fusion {
template<class T, int N> void prune(models::Mixture<T,N>&, double);
template<class T, int N> void cap(models::Mixture<T,N>&, std::size_t);
}

namespace brew::models {

// Specialize this for a new density, or pass a merger to Mixture::merge.
template<class T> struct MixtureMerge;


template <typename T, int N = Eigen::Dynamic>
class Mixture {
public:
    static constexpr bool fixed_capacity = (N != Eigen::Dynamic);

    using WeightVector = Eigen::Matrix<double, (fixed_capacity ? N : Eigen::Dynamic), 1>;

    using Storage = std::conditional_t<fixed_capacity,
        std::array<T, (N > 0 ? N : 1)>,
        std::vector<std::unique_ptr<T>>>;

    Mixture() {
        if constexpr (fixed_capacity) {
            weights_.setZero();
        }
    }

    void reduce(double prune_threshold, double merge_threshold, std::size_t capacity,
                double rotation_threshold = 1.5707963267948966);

    void merge(double threshold, double rotation_threshold = 1.5707963267948966) {
        MixtureMerge<T>::apply(*this, threshold, rotation_threshold);
    }

    template<class Merger>
    void merge(double threshold, Merger&& merger) {
        std::forward<Merger>(merger)(*this, threshold);
    }

    void merge_augmentation(T& result, const std::vector<std::size_t>& indices) const {
        if constexpr (requires { result.merge_augmentation({}); }) {
            std::vector<std::pair<const AugmentedState*, double>> sources;
            for (auto i : indices) sources.emplace_back(&component(i), weight(i));
            result.merge_augmentation(sources);
        }
    }

    [[nodiscard]] std::unique_ptr<Mixture<T, N>> clone() const {
        auto result = std::make_unique<Mixture<T, N>>();
        if constexpr (fixed_capacity) {
            for (std::size_t i = 0; i < size_; ++i) {
                result->components_[i] = components_[i];
                result->weights_(i) = weights_(i);
            }
            result->size_ = size_;
        } else {
            result->components_.reserve(components_.size());
            for (const auto& c : components_) {
                result->components_.push_back(c->clone_typed());
            }
            result->weights_ = weights_;
        }
        return result;
    }

    [[nodiscard]] std::size_t size() const {
        if constexpr (fixed_capacity) return size_;
        else return components_.size();
    }
    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] std::size_t length() const { return size(); }
    [[nodiscard]] static constexpr std::size_t max_size() {
        return fixed_capacity ? static_cast<std::size_t>(N)
                              : std::numeric_limits<std::size_t>::max();
    }

    [[nodiscard]] T& component(std::size_t i) {
        if constexpr (fixed_capacity) return components_[i];
        else return *components_[i];
    }
    [[nodiscard]] const T& component(std::size_t i) const {
        if constexpr (fixed_capacity) return components_[i];
        else return *components_[i];
    }

    [[nodiscard]] double weight(std::size_t i) const {
        return weights_(static_cast<Eigen::Index>(i));
    }

    [[nodiscard]] decltype(auto) weights() { return weights_live_view(); }
    [[nodiscard]] decltype(auto) weights() const { return weights_live_view(); }

    void add_component(std::unique_ptr<T> dist, double weight) {
        if constexpr (fixed_capacity) {
            assert(size_ < static_cast<std::size_t>(N) && "Mixture: capacity exceeded");
            components_[size_] = std::move(*dist);
            weights_(static_cast<Eigen::Index>(size_)) = weight;
            ++size_;
        } else {
            components_.push_back(std::move(dist));
            weights_.conservativeResize(weights_.size() + 1);
            weights_(weights_.size() - 1) = weight;
        }
    }

    void add_component(T dist, double weight) {
        if constexpr (fixed_capacity) {
            assert(size_ < static_cast<std::size_t>(N) && "Mixture: capacity exceeded");
            components_[size_] = std::move(dist);
            weights_(static_cast<Eigen::Index>(size_)) = weight;
            ++size_;
        } else {
            components_.push_back(std::make_unique<T>(std::move(dist)));
            weights_.conservativeResize(weights_.size() + 1);
            weights_(weights_.size() - 1) = weight;
        }
    }

    void add_components(const Mixture<T, N>& other) {
        for (std::size_t i = 0; i < other.size(); ++i) {
            if constexpr (fixed_capacity) {
                assert(size_ < static_cast<std::size_t>(N) && "Mixture: capacity exceeded");
                components_[size_] = other.components_[i];
                weights_(static_cast<Eigen::Index>(size_)) = other.weights_(static_cast<Eigen::Index>(i));
                ++size_;
            } else {
                components_.push_back(other.components_[i]->clone_typed());
                weights_.conservativeResize(weights_.size() + 1);
                weights_(weights_.size() - 1) = other.weights_(static_cast<Eigen::Index>(i));
            }
        }
    }

    void remove_component(std::size_t idx) {
        if constexpr (fixed_capacity) {
            for (std::size_t i = idx; i + 1 < size_; ++i) {
                components_[i] = std::move(components_[i + 1]);
                weights_(static_cast<Eigen::Index>(i)) =
                    weights_(static_cast<Eigen::Index>(i + 1));
            }
            --size_;
            weights_(static_cast<Eigen::Index>(size_)) = 0.0;
            components_[size_] = T{};
        } else {
            components_.erase(components_.begin() + static_cast<long>(idx));
            Eigen::VectorXd new_weights(weights_.size() - 1);
            for (Eigen::Index j = 0, k = 0; j < weights_.size(); ++j) {
                if (static_cast<std::size_t>(j) != idx) {
                    new_weights(k++) = weights_(j);
                }
            }
            weights_ = new_weights;
        }
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
        if constexpr (fixed_capacity) {
            for (std::size_t i = 0; i < size_; ++i) components_[i] = T{};
            size_ = 0;
            weights_.setZero();
        } else {
            components_.clear();
            weights_.resize(0);
        }
    }

    [[nodiscard]] Mixture<T, N> extract_mix(double threshold) const {
        Mixture<T, N> result;
        for (std::size_t i = 0; i < size(); ++i) {
            if (weight(i) >= threshold) {
                if constexpr (fixed_capacity) {
                    result.add_component(T{components_[i]}, weight(i));
                } else {
                    result.add_component(components_[i]->clone_typed(), weight(i));
                }
            }
        }
        return result;
    }

    [[nodiscard]] const Storage& components() const { return components_; }
    [[nodiscard]] Storage& components() { return components_; }

private:
    decltype(auto) weights_live_view() {
        if constexpr (fixed_capacity) {
            return weights_.head(static_cast<Eigen::Index>(size_));
        } else {
            return (weights_);
        }
    }
    decltype(auto) weights_live_view() const {
        if constexpr (fixed_capacity) {
            return weights_.head(static_cast<Eigen::Index>(size_));
        } else {
            return (weights_);
        }
    }

    Storage components_{};
    WeightVector weights_{};
    std::size_t size_ = 0;
};

}



#include "brew/utils/fusion/prune.hpp"
#include "brew/utils/fusion/cap.hpp"
namespace brew::models {
template<class T, int N>
void Mixture<T,N>::reduce(double prune_threshold, double merge_threshold,
                         std::size_t capacity, double rotation_threshold) {
    utils::fusion::prune(*this, prune_threshold);
    if constexpr (requires { MixtureMerge<T>::cap_before_merge; }) {
        if constexpr (MixtureMerge<T>::cap_before_merge) utils::fusion::cap(*this, capacity);
    }
    merge(merge_threshold, rotation_threshold);
    utils::fusion::cap(*this, capacity);
}
}
