# Bayesian Recursive Estimation Workspace (BREW)
---

A header-heavy C++ library for single-target and multi-target tracking, supporting both point and extended targets. Built on Eigen and C++20.

## Layout

BREW separates state estimation from multi-object management:

- **Densities** hold one object's state and parameters: Gaussian, GGIW, template pose, spherical state, or a trajectory.
- **Single-object filters** own prediction, measurement likelihoods, correction, gating, missed-detection updates, and any attached downstream parameter estimators.
- **Mixtures** hold weighted densities and expose merging and reduction. Each density supplies its appropriate merge rule.
- **RFS algorithms** manage births, survival, associations, weights, cardinality, and extraction. Each algorithm has one implementation templated on its density.

```text
include/brew/
  models/               densities, Mixture, augmentations, behavior models
    merge/              density-specific moment matching
  dynamics/             motion models
  filters/              single-object filters, IntentEstimator, IntentFilter
  multiObjectTrackers/  PHD, CPHD, GLMB, JGLMB, MBM, PMBM
  utils/                assignment, clustering, fusion, registration, metrics,
                        spherical geometry, sampling, optional plotting
```

Folders and C++ namespaces follow the same roles. Each RFS algorithm is templated
on its density; there are no per-density RFS packages or compatibility aliases.
Include the density's merge policy alongside its filter and RFS algorithm.
Compiled sources follow the same layout under `src/brew/`.

## Construction and ownership

```cpp
#include <brew/models/merge/ggiw.hpp>
#include <brew/filters/ggiw_ekf.hpp>
#include <brew/multiObjectTrackers/phd.hpp>
#include <brew/dynamics/single_integrator.hpp>

using Density = brew::models::GGIW<double, 4, 2>;
brew::dynamics::SingleIntegrator<double, 2> motion;
brew::filters::GGIWEKF<double, 4, 2> filter(motion);
filter.set_process_noise(Eigen::Matrix4d::Identity() * 0.01);
filter.set_measurement_noise(Eigen::Matrix2d::Identity());
Eigen::Matrix<double, Eigen::Dynamic, 4> H = Eigen::Matrix<double, Eigen::Dynamic, 4>::Zero(2, 4);
H.leftCols<2>().setIdentity();
filter.set_measurement_jacobian(H);

brew::multiObjectTrackers::PHD<Density> phd(filter);
phd.set_prob_detection(0.9);
phd.set_prob_survive(0.99);
phd.set_clutter_rate(1.0);
phd.set_clutter_density(1e-4);

// Add the study's birth densities, then repeat for each observation frame.
auto birth = std::make_unique<brew::models::Mixture<Density>>();
birth->add_component(Density(8., 2., Eigen::Vector4d::Zero(),
    Eigen::Matrix4d::Identity(), 10., 4.*Eigen::Matrix2d::Identity()), 0.1);
phd.set_birth_model(std::move(birth));
Eigen::MatrixXd observations = Eigen::MatrixXd::Zero(2, 1);
phd.predict(0, 1.0);
phd.correct(observations);
phd.cleanup();
```

The filter constructor clones the motion model; the RFS constructor clones the
configured filter. Configure them before constructing the RFS, or replace its
filter explicitly with `set_filter`. A filter for a different density cannot be
passed to the constructor. `CPHD<Density>`, `GLMB<Density>`, `JGLMB<Density>`,
`MBM<Density>`, and `PMBM<Density>` use the same typed-filter contract, with their
respective birth/cardinality/hypothesis settings.

`PHD::step` combines preparation, prediction, correction, cleanup, and extraction.
The existing individual methods remain available. Observation preparation and
adaptive births are supplied by the single-object filter; the generic PHD contains
no spherical-coordinate, template-registration, firing, or intent equations.

Optional downstream state is stored by `models::AugmentedState`, shared by the
single-state and trajectory densities. It provides typed
`augmentation<State>()` / `set_augmentation(state)` access. The type-erased holder
shares immutable attachments across copies; updates replace the attachment, so
association branches remain independent. `TemplatePose` has no built-in intent fields.

`filters::IntentEstimator<Density>` implements the common HMM, using
native `models::PredictiveBehaviorModel<Density>` hypotheses. Wrap any compatible
`Filter<Density>` with `filters::IntentFilter<Density, Estimator>` and pass it
directly to the existing RFS. Spacecraft intent uses the same categorical state.
Known-object `posterior_history()` and `evidence_recursion()` use the
same native implementation for every density. There is no additional intent
association tracker or report-wrapper type.
The physical `TmEkf` computes firing reports independently and stores them in
`TemplatePose::firing()`, including when no intent wrapper is attached.

`Mixture<Density>::merge()` still dispatches through `MixtureMerge<Density>`.
Each primary merger passes its selected source indices and original intensity
weights to `mixture.merge_augmentation(merged_density, indices)`. The attachment
dispatches to `AugmentationMerge<State>`. For `CategoricalIntent`, labels are
aligned by name and probabilities are intensity-weighted arithmetic averages;
unmarked births contribute a uniform prior. This categorical approximation
preserves the current marginal, discards state/intent correlation, and starts the
next observation context from the merged physical state. Intent never changes
merge eligibility, physical moments, or intensity mass.

For a new user-defined density, inherit `AugmentedState` and preserve it on copies.
Its custom merger must call `merge_augmentation` with the same sources and weights
used for the physical merge. The default one-to-one batch prediction preserves
attachments; custom branching/mixing predictions must copy parent attachments or
merge them with their own primary transition weights. These are generic data
propagation hooks, not intent calculations. Only one typed attachment is active
per density; a combined state can hold multiple downstream summaries if needed.

For template-pose intent, `models::IntentHypothesis<Pose>` pairs a label with a
native behavior model or native log-likelihood function. Ownership is handled by
the library; no helper factory or explicit shared pointer is needed. The model's
`log_evidence(current, previous, dt)` evaluates its predictive behavior using the
full pose-density pair. The existing `IntentEstimator` applies those contributions
to the HMM; it does not dispatch on model names or types. Compose it outside the
primary filter before constructing the PHD:

```cpp
brew::filters::TmEkf<> primary; // configure dynamics/noise/geometry first
namespace models = brew::models;
using Pose = models::TemplatePose<>;
const models::ImpulsePrior impulse{0.2, 0.01};
auto motion = std::make_shared<brew::dynamics::ClohessyWiltshire<>>(mean_motion);
motion->set_impulse_model(brew::dynamics::ImpulseModel(
    impulse, Eigen::Vector3d::UnitX(), {3, 4, 5}, true));
primary.set_dynamics(motion);
brew::filters::IntentEstimator<Pose> intent({
    {"coast", models::PointingFiringModel(models::PointingRule::Fixed, impulse,
                                   Eigen::Vector3d::UnitX(), 1., 0.)},
    {"approach", models::PointingFiringModel(models::PointingRule::Radial, impulse,
                                      Eigen::Vector3d::UnitX(), -1.)}
});
brew::filters::IntentFilter<Pose, decltype(intent)> filter(primary, intent);
brew::multiObjectTrackers::PHD<Pose> phd(filter);
```

`ImpulsePrior` defines the shared bound and physical firing probability once.
The dynamics own an `ImpulseModel` configured with this prior. Behavior models
inherit its probability unless a conditional override is supplied (zero
for coast). `TmEkf` uses its physical prior to compute the firing report;
`IntentEstimator` has no impulse parameters or firing-posterior methods.
Consumers copy the immutable
configuration; complete setup before wrapping the primary filter.

For a custom native likelihood, pass a function with signature
`double(const Pose& current, const Pose* previous, double dt)` directly:
`models::IntentHypothesis<Pose>{"custom", my_log_likelihood}`. It returns the joint **log**
likelihood; `previous` can be null. The library adapts it to the behavior-model
interface. Keep per-object mutable state in the density, not in the shared function.
Python selects exposed native model constructors rather than Python callbacks.

`IntentFilter` is a `Filter<Pose>` that delegates the primary filter operations
and applies HMM prediction/correction to each density's own context. The intent
update follows the primary correction on each association branch and never
changes its likelihood. A primary filter's shared branch-evidence adjustment may
refresh the posterior from the same saved prior before reduction, without a
second HMM transition. Missed observations supply no likelihood evidence.
Finalization does not update intent or associate reports. Primary filters contain
no intent estimator; configure them before wrapping.

Reduction uses geometric criteria and intensity weights. Intent labels do not
block geometric merges; categorical probabilities are aligned by label and
averaged by intensity. Firing moments and mixture-CDF quantiles are marginalized
as well. Intent history is stored in each density with a bounded immutable pose
snapshot, not in a separate object tracker. Thus moving intent before reduction
preserves tracking while permitting different downstream intent probabilities.

`PointingFiringModel` implements the dissertation's original likelihoods. Its
pointing rule, firing probability, uncertainty, and gain/holding priors are model
configuration supplied by the application. Additional native families implement
the same contract without adding tracker layers. Models are immutable shared
configuration; object-specific history remains in the existing densities and
estimator. No trajectory hypotheses or extra latent state are included.

## Constructor configuration

`filters::TmEkf<>::Config` and `multiObjectTrackers::PHD<Density>::Config`
are plain aggregates accepted by the existing classes. Configure application
values in the calling project:

```cpp
brew::filters::TmEkf<> primary({
    .dynamics = motion,
    .template_library = library,
    .icp = registration,
    .process_noise = Q,
    .measurement_noise = R,
    .measurement_jacobian = H,
    .rotation_process_noise = Q_rotation,
});
brew::filters::IntentFilter<Pose, decltype(intent)> wrapped(primary, intent);
brew::multiObjectTrackers::PHD<Pose> phd(wrapped, phd_settings);
```

The fields are named after their setters. Unspecified optional fields keep the
existing constructor defaults; zero is an explicit value. Construction invokes
the same setters and validation. The setter API remains available afterward.
`TmEkf::Config` includes the dynamics, registration/library handles, noise and
measurement Jacobian, plus its surface and pose-observation options. Native
measurement Jacobians may be matrices or callable functions as with the setter.
`PHD::Config` includes detection/survival/clutter settings, history and reduction
settings, clustering, `birth_model`, and `intensity`. The last two are borrowed
mixture pointers consumed during construction and cloned into the PHD; their
sources must remain alive through the constructor call. No configuration object
is retained by the estimator. Model handles keep their existing shared ownership.
Application hypotheses, noise matrices, and study settings belong in the journal,
not in library presets.

## Impulse dynamics

`dynamics::ImpulseModel` describes a Bernoulli/uniform velocity jump at the start
of a prediction interval. Attach it to an existing motion model with
`motion.set_impulse_model(model)`. The motion model owns the immutable impulse
configuration; copies and clones retain it. Filters read the same prior from
their dynamics. The old `TmEkf` impulse setters have been removed.

The model propagates the jump through selected columns of the motion transition
matrix and returns its mean shift, covariance, and attitude derivative. For
nonlinear motion this is a first-order response about the prior state. The
probability is per prediction interval; changing the interval does not rescale it.

The common filter prediction splits each mixture component into no-fire/fire
branches while preserving its total intensity and downstream augmentation.
Gaussian EKF/UKF and Euclidean extended-object filters such as GGIWEKF use this
operation with tracking-frame directions. A 2D constant-velocity state, for
example, uses an axis `[1, 0]` and velocity indices `[2, 3]`. Default indices
`[3, 4, 5]` describe a 3D position/velocity state.

`body_frame=true` requires an attitude-aware filter. `TmEkf` supplies attitude and
attitude covariance and applies the returned cross-covariance terms. It retains
pose-measurement conditioning and the bounded-magnitude firing posterior; those
are inference operations, not dynamics. Its physical predictions do not depend
on the intent labels or probabilities.

Use `filter.predict_mixture(dt, mixture)` (or the RFS prediction) for stochastic
motion. Single-density prediction cannot represent both alternatives. Other
state representations must implement `predict_impulse_branch` in their filter;
the default rejects trajectory representations and unsupported densities.
Body-frame use without attitude, and combining this model with `ImmTmEkf`'s
separate mode-transition model, are explicitly rejected.

## Adding a density

Provide a value type with `Vector`, `mean()`, `is_extended()`, and `clone_typed()`;
implement `filters::Filter<Density>` for its prediction, correction likelihood,
gating, and cloning. Override the optional missed-detection, report, or batch
operations when the observation model needs them. Trajectory densities expose
`current()` to identify their current single-object state.

Specialize `models::MixtureMerge<Density>::apply(mixture, threshold,
rotation_threshold)` with the density's compatible merge rule. Existing families
retain their established numerical merge implementations. A mixture can also
accept an explicit merge callable with `mixture.merge(threshold, merger)`.
`mixture.reduce(prune_threshold, merge_threshold, capacity)` preserves the PHD
mass through merging and applies pruning and capping. Mixture weights are
intensities; they are not automatically normalized to sum to one.

The filter interface uses virtual dispatch. Fixed state dimensions and fixed
mixture capacity remain supported, but this API is not allocation-free: filters,
association hypotheses, cloning, and some batch operations allocate. A hardware
implementation should assess those operations against its specific constraints.

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

Install Eigen 3.4 development headers before configuring. The default build
compiles the library and desktop measurement sampler without downloading source.
Tests require installed GoogleTest and fetch EigenRand. Optional plotting fetches
matplot++; downloads stay in the build directory. Key options:

| Option | Default | Effect |
|---|---|---|
| `BREW_TARGET` | `desktop` | `desktop` (full library + sampling/plotting) or `hardware` (drops desktop modules, exceptions off, `BREW_ASSERT` instead of throw) |
| `BREW_ENABLE_PLOTTING` | `OFF` | matplot++ plotting (desktop only) |
| `BREW_BUILD_EXAMPLES` | `OFF` | Standalone C++ example |
| `BREW_BUILD_TESTS` | `OFF` | GoogleTest suite |


## Testing

```bash
cmake -S . -B build-tests -DBREW_BUILD_TESTS=ON
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```

Role-header and native verification without MATLAB is available
via `docker/` — see `docker/README.md`.

