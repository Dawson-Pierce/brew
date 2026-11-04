#include "brew/filters/intent_estimator.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "brew/models/pointing_firing_model.hpp"
#include "brew/models/merge/template_pose.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include "brew/filters/tm_ekf.hpp"
#include "brew/filters/intent_filter.hpp"
#include "brew/dynamics/clohessy_wiltshire.hpp"
#include <cassert>
#include <iostream>

using Pose = brew::models::TemplatePose<>;
using Mixture = brew::models::Mixture<Pose>;
using Estimator = brew::filters::IntentEstimator<brew::models::TemplatePose<>>;
using Hypothesis = brew::models::IntentHypothesis<brew::models::TemplatePose<>>;
using Pointing = brew::models::PointingRule;
using Behavior = brew::models::PredictiveBehaviorModel<brew::models::TemplatePose<>>;
using PointingFiring = brew::models::PointingFiringModel;
static_assert(std::is_same_v<Estimator, brew::filters::IntentEstimator<Pose>>);
using IntentFilter = brew::filters::IntentFilter<Pose, Estimator>;

Hypothesis hypothesis(std::string name, Pointing rule, Eigen::Vector3d axis, double sign, double probability) {
    auto model = std::make_shared<PointingFiring>();
    model->pointing = rule; model->axis = axis; model->radial_sign = sign;
    model->firing_probability = probability; model->impulse_max = .2;
    return {std::move(name), model};
}

// Test-only evidence probe, not an application/physics hypothesis. It verifies
// that an unrelated native implementation receives the density pair and dt.
class EvidenceProbe final : public Behavior {
public:
    void validate() const override {}
    std::vector<double> log_evidence(const Pose& current, const Pose* previous, double dt) const override {
        return previous ? std::vector<double>{-(current.position() - previous->position()).norm() / dt, -.5}
                        : std::vector<double>{0.};
    }
};

std::vector<Hypothesis> test_hypotheses() {
    return {hypothesis("idle", Pointing::Fixed, Eigen::Vector3d::UnitX(), 1., 0.),
            hypothesis("inward", Pointing::Radial, Eigen::Vector3d::UnitX(), -1., .01),
            hypothesis("outward", Pointing::Radial, Eigen::Vector3d::UnitX(), 1., .01),
            hypothesis("axis-y", Pointing::Fixed, Eigen::Vector3d::UnitY(), 1., .01),
            hypothesis("tangent", Pointing::Tangential, Eigen::Vector3d::UnitZ(), 1., .01)};
}

Pose pose(double x, int template_id = 0) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(6);
    mean[0] = x;
    Pose result(mean, Eigen::MatrixXd::Identity(9, 9) * 0.01, template_id, {0, 1, 2});
    result.mark_pca_aligned();
    return result;
}

void close(double a, double b, double tolerance = 1e-11) { assert(std::abs(a - b) <= tolerance); }

// Deterministic primary filter isolates the wrapper's scheduling and proves that
// downstream evidence cannot change returned measurement likelihoods or weights.
class PrimaryProbe final : public brew::filters::Filter<Pose> {
public:
    std::unique_ptr<Filter<Pose>> clone() const override { return std::make_unique<PrimaryProbe>(*this); }
    Pose predict(double dt, const Pose& previous) const override {
        auto result = previous; result.mean()[0] += dt; return result;
    }
    CorrectionResult correct(const MeasVector& measurement, const Pose& predicted) const override {
        auto result = predicted; result.mean()[0] = measurement[0];
        if (measurement.size() > 1) result.set_impulse_evidence(measurement[1], 1e-5);
        return {result, .25};
    }
    LogCorrectionResult correct_with_log(const MeasVector& measurement, const Pose& predicted) const override {
        return {correct(measurement, predicted), -42.};
    }
    CorrectionResult correct_report(const Pose& report, const Pose& predicted) const override {
        return correct(report.position(), predicted);
    }
    double gate(const MeasVector&, const Pose&) const override { return .125; }
    bool log_domain_correction() const override { return true; }
    void adjust_log_weights(std::vector<std::unique_ptr<Pose>>& states,
            const std::vector<double>&, std::vector<double>& weights) const override {
        for (std::size_t i = 0; i < states.size(); ++i) {
            weights[i] += .75;
            states[i]->set_impulse_branch_evidence(-2., -1.);
        }
    }
};

int main() {
    const brew::models::ImpulsePrior impulse(.2, .01);
    Estimator configured({
        {"idle", PointingFiring(Pointing::Fixed, impulse, Eigen::Vector3d::UnitX(), 1., 0.)},
        {"inward", PointingFiring(Pointing::Radial, impulse, Eigen::Vector3d::UnitX(), -1.)},
        {"outward", PointingFiring(Pointing::Radial, impulse)},
        {"axis-y", PointingFiring(Pointing::Fixed, impulse, Eigen::Vector3d::UnitY())},
        {"tangent", PointingFiring(Pointing::Tangential, impulse, Eigen::Vector3d::UnitZ())}
    });
    brew::filters::TmEkf<> configured_primary;
    auto motion = std::make_shared<brew::dynamics::ClohessyWiltshire<>>(.001);
    motion->set_impulse_model(brew::dynamics::ImpulseModel(impulse, Eigen::Vector3d::UnitX(), {3,4,5}, true));
    configured_primary.set_dynamics(motion);
    const auto missed_birth = configured_primary.missed_detection(pose(100), .8, 1.);
    assert(std::isnan(missed_birth[0].distribution.firing().impulse_probability));
    Estimator estimator;
    estimator.hypotheses = test_hypotheses();
    auto first = pose(100);
    estimator.correct(first, nullptr);
    close(first.augmentation<brew::models::CategoricalIntent>().probabilities.sum(), 1);
    assert(first.augmentation<brew::models::CategoricalIntent>().probabilities.size() == 5);
    auto cloned = first.clone_typed();
    assert(cloned->augmentation<brew::models::CategoricalIntent>().probabilities == first.augmentation<brew::models::CategoricalIntent>().probabilities);
    auto second = first;
    second.set_impulse_branch(0, 0);
    second.set_impulse_evidence(0.1, 1e-5);
    configured_primary.update_firing_report(second);
    estimator.correct(second, &first);
    auto shared_first = pose(100), shared_second = pose(100);
    shared_second.set_impulse_branch(0, 0);
    shared_second.set_impulse_evidence(.1, 1e-5);
    configured.correct(shared_first, nullptr);
    configured_primary.update_firing_report(shared_second);
    configured.correct(shared_second, &shared_first);
    const auto& shared_mark = shared_second.augmentation<brew::models::CategoricalIntent>();
    const auto& old_mark = second.augmentation<brew::models::CategoricalIntent>();
    assert(shared_mark.probabilities == old_mark.probabilities);
    assert(shared_second.firing().impulse_probability == second.firing().impulse_probability);
    assert(shared_second.firing().impulse_magnitude == second.firing().impulse_magnitude);
    assert(shared_second.firing().impulse_lower == second.firing().impulse_lower);
    assert(shared_second.firing().impulse_upper == second.firing().impulse_upper);
    assert(second.firing().impulse_probability > 0.99);
    close(second.firing().impulse_magnitude, 0.1, 1e-6);
    assert(second.firing().impulse_lower < 0.1 && second.firing().impulse_upper > 0.1);
    const auto same_interval = brew::utils::firing_detail::firing_interval(second.firing().firing_components);
    close(same_interval[0], second.firing().impulse_lower);
    close(same_interval[1], second.firing().impulse_upper);
    assert(second.mean() == first.mean() && second.covariance() == first.covariance());
    assert(second.augmentation<brew::models::CategoricalIntent>().probabilities[0] < first.augmentation<brew::models::CategoricalIntent>().probabilities[0]);
    const auto sequence = estimator.posterior_history({first, second});
    assert((sequence.row(1).transpose() - second.augmentation<brew::models::CategoricalIntent>().probabilities).norm() < 1e-12);

    // Labels do not select behavior; renaming every policy leaves likelihoods unchanged.
    auto renamed = estimator;
    for (std::size_t i = 0; i < renamed.hypotheses.size(); ++i)
        renamed.hypotheses[i].name = "custom-" + std::to_string(i);
    for (std::size_t i = 0; i < renamed.hypotheses.size(); ++i)
        assert(estimator.hypotheses[i].model->log_evidence(second, &first, estimator.dt)
            == renamed.hypotheses[i].model->log_evidence(second, &first, renamed.dt));
    auto always = hypothesis("certain-burn", Pointing::Fixed, Eigen::Vector3d::UnitX(), 1., 1.);
    renamed.hypotheses = {always};
    close(always.model->log_evidence(second, &first, renamed.dt)[1], brew::utils::firing_detail::impulse_log_evidence(.1, 1e-5, .2)[1]);
    assert(renamed.posterior_history({pose(100)})(0, 0) == 1.);

    // Mix implementations without changing the HMM or adding a tracker.
    auto neutral = std::make_shared<PointingFiring>();
    neutral->use_direction = false; neutral->use_impulse = false; neutral->holding_probability = 0.;
    Estimator generic;
    generic.dt = 2.;
    generic.hypotheses = {{"probe", std::make_shared<EvidenceProbe>()}, {"neutral", neutral}};
    auto origin = pose(100), moved = pose(102);
    generic.correct(origin, nullptr);
    generic.correct(moved, &origin);
    close(moved.augmentation<brew::models::CategoricalIntent>().probabilities[0], std::exp(-1.5) / (1. + std::exp(-1.5)));
    assert(moved.mean() == pose(102).mean());
    auto gap = pose(102);
    generic.correct(gap, &origin, false);
    close(gap.augmentation<brew::models::CategoricalIntent>().probabilities[0], .5);
    auto copied = generic;
    auto independent = pose(102);
    copied.correct(independent, &origin);
    assert(independent.augmentation<brew::models::CategoricalIntent>().probabilities == moved.augmentation<brew::models::CategoricalIntent>().probabilities);

    PrimaryProbe generic_filter;
    IntentFilter generic_wrapper(generic_filter, generic);
    brew::multiObjectTrackers::PHD<Pose> generic_phd(generic_wrapper);
    Mixture generic_mixture;
    generic_mixture.add_component(pose(100), 1.);
    generic_phd.set_intensity(generic_mixture.clone());
    auto initial = generic_wrapper.correct(Eigen::VectorXd::Constant(1, 100.), pose(100)).distribution;
    auto predicted = generic_wrapper.predict(2., initial);
    close(predicted.position()[0], 102.);
    const auto* snapshot = dynamic_cast<const brew::models::IntentDensityObservation<Pose>*>(
        predicted.augmentation<brew::models::CategoricalIntent>().previous_observation.get());
    assert(snapshot && snapshot->density.augmentation<brew::models::CategoricalIntent>().probabilities.size() == 0);
    auto measured = Eigen::VectorXd::Constant(1, 104.).eval();
    auto primary_result = generic_filter.correct_with_log(measured, predicted);
    auto branch_result = generic_wrapper.correct_with_log(measured, predicted);
    assert(primary_result.first.likelihood == branch_result.first.likelihood && primary_result.second == branch_result.second);
    assert(primary_result.first.distribution.mean() == branch_result.first.distribution.mean());
    assert(primary_result.first.distribution.covariance() == branch_result.first.distribution.covariance());
    close(branch_result.first.distribution.augmentation<brew::models::CategoricalIntent>().probabilities[0], std::exp(-2.5) / (1. + std::exp(-2.5)));
    auto missed = generic_wrapper.missed_detection(predicted, .8, .7);
    close(missed[0].weight, .14);
    assert(missed[0].distribution.augmentation<brew::models::CategoricalIntent>().probabilities == predicted.augmentation<brew::models::CategoricalIntent>().probabilities);
    assert(!missed[0].distribution.augmentation<brew::models::CategoricalIntent>().observed);
    auto after_gap = generic_wrapper.predict(2., missed[0].distribution);
    assert(!after_gap.augmentation<brew::models::CategoricalIntent>().previous_observation);
    auto reacquired = generic_wrapper.correct(measured, after_gap);
    close(reacquired.distribution.augmentation<brew::models::CategoricalIntent>().probabilities[0], .5);

    // All branch histories live in densities: cloning filters or evaluating a
    // sibling must not mutate another branch or apply the HMM transition twice.
    auto clone_a = generic_wrapper.clone(), clone_b = generic_wrapper.clone();
    auto next_a = clone_a->correct(Eigen::VectorXd::Constant(1, 102.), predicted);
    auto next_b = clone_b->correct(Eigen::VectorXd::Constant(1, 104.), predicted);
    close(next_a.distribution.augmentation<brew::models::CategoricalIntent>().probabilities[0], std::exp(-1.5) / (1. + std::exp(-1.5)));
    close(next_b.distribution.augmentation<brew::models::CategoricalIntent>().probabilities[0], std::exp(-2.5) / (1. + std::exp(-2.5)));
    auto next_prediction = generic_wrapper.predict(2., next_b.distribution);
    close(next_prediction.augmentation<brew::models::CategoricalIntent>().probabilities[0], .9 * next_b.distribution.augmentation<brew::models::CategoricalIntent>().probabilities[0]
                                                  + .1 * next_b.distribution.augmentation<brew::models::CategoricalIntent>().probabilities[1]);
    std::vector<std::unique_ptr<Pose>> states;
    states.push_back(next_b.distribution.clone_typed());
    std::vector<double> weights = {-3.};
    generic_wrapper.adjust_log_weights(states, {1.}, weights);
    close(weights[0], -2.25);
    assert(states[0]->augmentation<brew::models::CategoricalIntent>().probabilities == next_b.distribution.augmentation<brew::models::CategoricalIntent>().probabilities);
    brew::models::Mixture<Pose, 4> fixed;
    fixed.add_component(initial, 1.);
    generic_wrapper.predict_mixture(2., fixed);
    assert(fixed.component(0).augmentation<brew::models::CategoricalIntent>().probabilities == predicted.augmentation<brew::models::CategoricalIntent>().probabilities);
    // Intent is already present before cleanup, and cleanup must not update it.
    generic_phd.set_prob_detection(1.); generic_phd.set_intensity(generic_mixture.clone());
    generic_phd.correct(Eigen::MatrixXd::Constant(1, 1, 100.));
    assert(generic_phd.intensity().component(0).augmentation<brew::models::CategoricalIntent>().probabilities.size() == 2);

    // Weighted categorical merge and total variance; no discarded intent mass.
    auto a = first, b = first;
    auto state = first.augmentation<brew::models::CategoricalIntent>();
    state.probabilities << 1, 0, 0, 0, 0;
    brew::models::FiringEstimate firing;
    firing.impulse_probability = 0.2;
    firing.impulse_magnitude = 0.02;
    firing.impulse_std = 0.01;
    a.set_augmentation(state);
    a.set_firing(firing);
    state.probabilities << 0, 1, 0, 0, 0;
    firing.impulse_probability = 0.8;
    firing.impulse_magnitude = 0.08;
    firing.impulse_std = 0.02;
    b.set_augmentation(state);
    b.set_firing(firing);
    Mixture mixture;
    mixture.add_component(a, 0.25);
    mixture.add_component(b, 0.75);
    brew::models::merge(mixture, 4);
    assert(mixture.size() == 1);
    close(mixture.weight(0), 1);
    const auto& merged = mixture.component(0).augmentation<brew::models::CategoricalIntent>();
    close(merged.probabilities[0], 0.25);
    close(merged.probabilities[1], 0.75);
    close(mixture.component(0).firing().impulse_probability, 0.65);
    close(mixture.component(0).firing().impulse_magnitude, 0.065);
    close(mixture.component(0).firing().impulse_std * mixture.component(0).firing().impulse_std,
          0.25 * (0.0001 + 0.0004) + 0.75 * (0.0004 + 0.0064) - 0.065 * 0.065);
    assert(std::isnan(mixture.component(0).firing().impulse_lower));
    mixture = Mixture{};
    b.set_impulse_branch(1);
    mixture.add_component(a, 0.25);
    mixture.add_component(b, 0.75);
    brew::models::merge(mixture, 4);
    assert(mixture.size() == 2);

    // Intent labels never block a geometric merge; reorder probabilities by name.
    mixture = Mixture{};
    auto alternate = a;
    state = a.augmentation<brew::models::CategoricalIntent>();
    std::swap(state.objectives[0], state.objectives[1]);
    alternate.set_augmentation(state);
    mixture.add_component(pose(100), 0.8);
    mixture.add_component(a, 0.1);
    mixture.add_component(alternate, 0.1);
    brew::models::merge(mixture, 4);
    assert(mixture.size() == 1);
    close(mixture.component(0).augmentation<brew::models::CategoricalIntent>().probabilities[0], .26);
    close(mixture.component(0).augmentation<brew::models::CategoricalIntent>().probabilities[1], .26);

    // Merged credible intervals come from the mixture CDF, not averaged bounds.
    auto low = pose(100), high = pose(100);
    auto low_mark = brew::utils::firing_detail::firing_prior(1., .1);
    auto high_mark = brew::utils::firing_detail::firing_prior(1., .2);
    low.set_firing(low_mark); high.set_firing(high_mark);
    Mixture intervals; intervals.add_component(low, .5); intervals.add_component(high, .5);
    brew::models::merge(intervals, 4);
    close(intervals.component(0).firing().impulse_lower, .025 / 7.5);
    close(intervals.component(0).firing().impulse_upper, .19);

    // Native PHD cleanup only merges existing marks; it cannot create an HMM update.
    for (std::size_t i = 0; i < mixture.size(); ++i) mixture.component(i).clear_augmentation();
    brew::filters::TmEkf<> filter;
    IntentFilter wrapper(filter, estimator);
    brew::multiObjectTrackers::PHD<brew::models::TemplatePose<>, Eigen::Dynamic> plain;
    brew::multiObjectTrackers::PHD<Pose> marked(wrapper);
    plain.set_intensity(mixture.clone());
    marked.set_intensity(mixture.clone());
    plain.cleanup(); marked.cleanup();
    assert(plain.intensity().size() == marked.intensity().size());
    for (std::size_t i = 0; i < plain.intensity().size(); ++i) {
        close(plain.intensity().weight(i), marked.intensity().weight(i));
        assert(plain.intensity().component(i).mean() == marked.intensity().component(i).mean());
        assert(marked.intensity().component(i).augmentation<brew::models::CategoricalIntent>().probabilities.size() == 0);
    }
    std::cout << "Intent recursion, variable object count, firing, merge, and PHD integration passed\n";
}
