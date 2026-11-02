#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brew::models {

// A density may carry a downstream state without changing its physical type.
// Published attachments are immutable; replacing one cannot mutate sibling branches.
class Augmentation {
public:
    using Sources = std::vector<std::pair<const Augmentation*, double>>;
    virtual ~Augmentation() = default;
    virtual std::shared_ptr<const Augmentation> merge(const Sources& sources) const = 0;
};

template<class State> struct AugmentationMerge;

template<class State>
class DensityAugmentation final : public Augmentation {
public:
    explicit DensityAugmentation(State state) : value(std::move(state)) {}
    const State value;

    std::shared_ptr<const Augmentation> merge(const Sources& sources) const override {
        std::vector<std::pair<const State*, double>> states;
        for (const auto& [source, weight] : sources) {
            const auto* typed = dynamic_cast<const DensityAugmentation*>(source);
            if (source && !typed) throw std::invalid_argument("Cannot merge different augmentation types");
            states.emplace_back(typed ? &typed->value : nullptr, weight);
        }
        return std::make_shared<DensityAugmentation>(AugmentationMerge<State>::apply(states));
    }
};

class AugmentedState {
public:
    template<class State> const State& augmentation() const {
        if (!augmentation_) { static const State empty{}; return empty; }
        const auto* typed = dynamic_cast<const DensityAugmentation<State>*>(augmentation_.get());
        if (!typed) throw std::invalid_argument("Density has a different augmentation type");
        return typed->value;
    }
    template<class State> void set_augmentation(State state) {
        augmentation_ = std::make_shared<DensityAugmentation<State>>(std::move(state));
    }
    void clear_augmentation() { augmentation_.reset(); }
    void copy_augmentation_from(const AugmentedState& source) { augmentation_ = source.augmentation_; }

    // Weights and source membership are supplied exclusively by the primary merger.
    void merge_augmentation(const std::vector<std::pair<const AugmentedState*, double>>& sources) {
        Augmentation::Sources marks;
        std::shared_ptr<const Augmentation> prototype;
        for (const auto& [source, weight] : sources) {
            marks.emplace_back(source->augmentation_.get(), weight);
            if (!prototype) prototype = source->augmentation_;
        }
        augmentation_ = prototype ? prototype->merge(marks) : nullptr;
    }

private:
    std::shared_ptr<const Augmentation> augmentation_;
};

} // namespace brew::models
