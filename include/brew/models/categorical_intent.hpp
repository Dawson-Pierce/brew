#pragma once

#include "brew/models/base_single_model.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <optional>
#include "brew/utils/assert.hpp"
#include <string>
#include <type_traits>
#include <vector>

namespace brew::models {

struct IntentObservation { virtual ~IntentObservation() = default; };
template<class Dist> struct IntentDensityObservation final : IntentObservation {
    explicit IntentDensityObservation(const Dist& source) : density(source) { density.clear_augmentation(); }
    Dist density;
};

struct CategoricalIntent {
    std::vector<std::string> objectives;
    Eigen::VectorXd probabilities;
    Eigen::VectorXd predicted_probabilities;
    std::shared_ptr<const IntentObservation> previous_observation;
    double interval_dt = 0.0;
    bool observed = false;

    void validate() const {
        if (probabilities.size() != static_cast<int>(objectives.size()))
            throw std::invalid_argument("Intent probabilities do not match their labels");
        if (probabilities.size() && (!probabilities.allFinite() || (probabilities.array() < 0).any()
                || std::abs(probabilities.sum() - 1.) > 1e-10))
            throw std::invalid_argument("Intent probabilities must be normalized and nonnegative");
        for (std::size_t i = 0; i < objectives.size(); ++i) {
            if (objectives[i].empty()) throw std::invalid_argument("Intent labels cannot be empty");
            if (std::find(objectives.begin(), objectives.begin() + i, objectives[i]) != objectives.begin() + i)
                throw std::invalid_argument("Duplicate intent label");
        }
    }
};

template<> struct AugmentationMerge<CategoricalIntent> {
    static CategoricalIntent apply(const std::vector<std::pair<const CategoricalIntent*, double>>& sources) {
        CategoricalIntent result;
        double total = 0;
        for (const auto& [state, weight] : sources) {
            if (!std::isfinite(weight) || weight < 0) throw std::invalid_argument("Invalid augmentation weight");
            total += weight;
            if (!state || weight == 0) continue;
            state->validate();
            for (const auto& label : state->objectives)
                if (std::find(result.objectives.begin(), result.objectives.end(), label) == result.objectives.end())
                    result.objectives.push_back(label);
        }
        if (sources.size() == 1 && sources[0].first) return *sources[0].first;
        if (result.objectives.empty() || total <= 0) return result;
        result.probabilities = Eigen::VectorXd::Zero(result.objectives.size());
        for (const auto& [state, weight] : sources) {
            if (weight == 0) continue;
            const double q = weight / total;
            if (!state || state->probabilities.size() == 0) {
                result.probabilities.array() += q / result.probabilities.size();
                continue;
            }
            for (std::size_t j = 0; j < state->objectives.size(); ++j) {
                const auto target = std::find(result.objectives.begin(), result.objectives.end(), state->objectives[j]);
                result.probabilities[target - result.objectives.begin()] += q * state->probabilities[j];
            }
            result.observed = result.observed || state->observed;
        }
        // This is a categorical marginal, not a retained state/history mixture.
        // The next prediction snapshots the merged primary density.
        return result;
    }
};

} // namespace brew::models

namespace brew::models {

template<class Dist> class PredictiveBehaviorModel {
public:
    virtual ~PredictiveBehaviorModel() = default;
    virtual void validate() const = 0;
    virtual std::vector<double> log_evidence(const Dist& current, const Dist* previous, double dt) const = 0;
};

// A native likelihood function can be a hypothesis without a user-written class.
// The callable returns a joint log likelihood; mutable per-object state belongs
// to the density, not to the callable shared by association branches.
template<class Dist> class FunctionBehavior final : public PredictiveBehaviorModel<Dist> {
public:
    using Likelihood = std::function<double(const Dist&, const Dist*, double)>;
    explicit FunctionBehavior(Likelihood likelihood) : likelihood_(std::move(likelihood)) {}
    void validate() const override {
        if (!likelihood_) throw std::invalid_argument("Intent likelihood function is empty");
    }
    std::vector<double> log_evidence(const Dist& current, const Dist* previous, double dt) const override {
        return {likelihood_(current, previous, dt)};
    }
private:
    Likelihood likelihood_;
};

template<class Dist> struct IntentHypothesis {
    std::string name;
    std::shared_ptr<const PredictiveBehaviorModel<Dist>> model;

    IntentHypothesis() = default;
    IntentHypothesis(std::string name, std::shared_ptr<const PredictiveBehaviorModel<Dist>> model)
        : name(std::move(name)), model(std::move(model)) {}
    template<class Model> requires std::derived_from<Model, PredictiveBehaviorModel<Dist>>
    IntentHypothesis(std::string name, Model model)
        : IntentHypothesis(std::move(name), std::make_shared<const Model>(std::move(model))) {}
    template<class Function> requires std::is_invocable_r_v<double, Function, const Dist&, const Dist*, double>
    IntentHypothesis(std::string name, Function function)
        : IntentHypothesis(std::move(name), std::make_shared<const FunctionBehavior<Dist>>(std::move(function))) {}

    void validate() const {
        if (name.empty() || !model) throw std::invalid_argument("Intent needs a name and native behavior model");
        model->validate();
    }
};

// An uninformative native behavior is useful when a hypothesis has no evidence
// model yet. It contributes no evidence, leaving only the HMM prediction.
template<class Dist> class UninformativeBehavior final : public PredictiveBehaviorModel<Dist> {
public:
    void validate() const override {}
    std::vector<double> log_evidence(const Dist&, const Dist*, double) const override { return {}; }
};

} // namespace brew::models
