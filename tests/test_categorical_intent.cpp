#ifdef NDEBUG
#undef NDEBUG
#endif
#include "brew/filters/intent_estimator.hpp"
#include "brew/filters/intent_filter.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include <cassert>
#include <iostream>

// A user-defined density and filter: no Gaussian, pose, or spacecraft dependency.
struct Density : brew::models::AugmentedState {
    using Vector = Eigen::Vector2d;
    Vector x = Vector::Zero();
    Density() = default;
    explicit Density(Vector value) : x(std::move(value)) {}
    const Vector& mean() const { return x; }
    bool is_extended() const { return false; }
    auto clone_typed() const { return std::make_unique<Density>(*this); }
};
using State = brew::models::CategoricalIntent;
using Estimator = brew::filters::IntentEstimator<Density>;

namespace brew::models {
template<> struct MixtureMerge<Density> {
    template<int N> static void apply(Mixture<Density,N>& mix, double threshold, double) {
        if (threshold <= 0 || mix.size() < 2) return;
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        double weight = 0;
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            mean += mix.weight(i) * mix.component(i).mean(); weight += mix.weight(i);
            indices.push_back(i);
        }
        Density merged(mean / weight);
        mix.merge_augmentation(merged, indices);
        mix.clear(); mix.add_component(merged, weight);
    }
};
}

struct Primary : brew::filters::Filter<Density> {
    std::unique_ptr<Filter> clone() const override { return std::make_unique<Primary>(*this); }
    Density predict(double dt, const Density& previous) const override {
        return Density(previous.x + Eigen::Vector2d(dt, 0)); // constructs fresh state
    }
    CorrectionResult correct(const MeasVector& z, const Density& predicted) const override {
        return {Density(.5 * (z + predicted.x)), .25};
    }
    double gate(const MeasVector&, const Density&) const override { return 0.; }
    std::vector<WeightedDensity> missed_detection(const Density& prior, double pd, double weight) const override {
        return {{Density(prior.x), weight * (1 - pd)}};
    }
};
struct Evidence : brew::models::PredictiveBehaviorModel<Density> {
    double likelihood;
    explicit Evidence(double value) : likelihood(value) {}
    void validate() const override {}
    std::vector<double> log_evidence(const Density&, const Density*, double) const override {
        return {std::log(likelihood)};
    }
};
State mark(double first, bool reverse = false) {
    State state;
    state.objectives = reverse ? std::vector<std::string>{"b", "a"} : std::vector<std::string>{"a", "b"};
    state.probabilities = Eigen::Vector2d(first, 1-first);
    return state;
}
void near(double a, double b) { assert(std::abs(a-b) < 1e-12); }

int main() {
    // Both native model values and native functions construct named hypotheses.
    Estimator direct({{"a", Evidence(.8)},
        {"b", [](const Density& current, const Density* previous, double dt) {
            assert(current.x == Eigen::Vector2d(3,4));
            assert(previous && previous->x == Eigen::Vector2d(1,2));
            near(dt, 30.);
            return std::log(.2);
        }}});
    Density before(Eigen::Vector2d(1,2)), after(Eigen::Vector2d(3,4));
    auto observed = mark(.5); observed.observed = true;
    before.set_augmentation(observed);
    direct.prepare_prediction(before);
    direct.predict(before, 30.);
    direct.correct_branch(after, before);
    near(after.augmentation<State>().probabilities[0], .8);
    brew::models::IntentHypothesis<Density> empty{"empty",
        brew::models::FunctionBehavior<Density>::Likelihood{}};
    bool rejected = false;
    try { empty.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    Primary primary;
    Estimator estimator;
    estimator.hypotheses = {{"a", std::make_shared<Evidence>(.8)}, {"b", std::make_shared<Evidence>(.2)}};
    brew::filters::IntentFilter<Density, Estimator> wrapper(primary, estimator);
    Density initial(Eigen::Vector2d(1,2)); initial.set_augmentation(mark(.8));
    auto predicted = wrapper.predict(1., initial);
    assert(predicted.x == primary.predict(1., initial).x);
    near(predicted.augmentation<State>().probabilities[0], .74);
    auto updated = wrapper.correct(Eigen::Vector2d(3,4), predicted);
    auto plain = primary.correct(Eigen::Vector2d(3,4), predicted);
    assert(updated.distribution.x == plain.distribution.x && updated.likelihood == plain.likelihood);
    near(updated.distribution.augmentation<State>().probabilities[0], .74*.8/(.74*.8+.26*.2));
    auto missed = wrapper.missed_detection(predicted, .8, .7);
    near(missed[0].weight, .14);
    near(missed[0].distribution.augmentation<State>().probabilities[0], .74);
    near(wrapper.missed_detection(Density{}, .8, .7)[0].distribution.augmentation<State>().probabilities[0], .5);
    auto clone = updated.distribution.clone_typed();
    clone->set_augmentation(mark(.1));
    assert(clone->augmentation<State>().probabilities != updated.distribution.augmentation<State>().probabilities);

    brew::models::Mixture<Density, 4> mixture;
    Density other(Eigen::Vector2d(2,3)); other.set_augmentation(mark(.8, true));
    mixture.add_component(initial, 3.); mixture.add_component(other, 1.);
    mixture.merge(1.);
    assert(mixture.size() == 1); near(mixture.weight(0), 4.);
    near(mixture.component(0).augmentation<State>().probabilities[0], .65);
    assert(mixture.component(0).x == Eigen::Vector2d(1.25,2.25));
    wrapper.predict_mixture(1., mixture);
    near(mixture.component(0).augmentation<State>().probabilities[0], .62);

    // An unmarked birth contributes a uniform categorical prior, not zero mass.
    mixture.clear(); mixture.add_component(initial, 3.); mixture.add_component(Density{}, 1.);
    mixture.merge(1.); near(mixture.component(0).augmentation<State>().probabilities[0], .725);
    // Zero probability stays zero under evidence; zero transition probability is valid.
    estimator.objective_stay = 1.;
    brew::filters::IntentFilter<Density, Estimator> certain(primary, estimator);
    initial.set_augmentation(mark(1.));
    near(certain.correct(Eigen::Vector2d(0,0), certain.predict(1.,initial)).distribution.augmentation<State>().probabilities[0], 1.);
    brew::multiObjectTrackers::PHD<Density> rfs(wrapper);
    rfs.set_prob_detection(.8);
    rfs.intensity().add_component(initial, 1.);
    rfs.predict(0, 1.); rfs.correct(Eigen::MatrixXd(2,0)); rfs.cleanup();
    assert(rfs.intensity().size() == 1);
    near(rfs.intensity().component(0).augmentation<State>().probabilities.sum(), 1.);
    std::cout << "Generic categorical augmentation regression passed\n";
}
