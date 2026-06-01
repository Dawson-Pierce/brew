# Bayesian Recursive Estimation Workspace (BREW)
---

A header-heavy C++ library for single-target and multi-target tracking, supporting both point and extended targets. Built on Eigen and C++20.

## Layout

The library is organized into **flat, per-model-type packages** plus a shared
core and standalone subsystems. Each model package owns its data structure, its
single-object filters, and (the entry points to) the multi-object filters usable
with it. Include a package's umbrella header to pull in its whole stack:

```cpp
#include <brew/gaussian/gaussian.hpp>   // Gaussian model + EKF + PHD/CPHD/GLMB/...
```

```
include/brew/
  shared/        base classes (filter_base, rfs_base, base_single_model),
                 generic containers (mixture, bernoulli, trajectory), the generic
                 RFS templates (shared/multi_target_generic/), fusion primitives
  dynamics/  clustering/  template_matching/  assignment/  metrics/   (standalone subsystems)
  gaussian/  ggiw/  ggiw_orientation/  iggiw/  template_pose/           (model packages)
  trajectory_gaussian/  trajectory_ggiw/  trajectory_ggiw_orientation/
  trajectory_iggiw/  trajectory_template_pose/
  desktop/       plotting + sampling (desktop builds only)
```

Each package is `#include`-able via `brew/<pkg>/<pkg>.hpp` and builds as its own
CMake library `brew_pkg_<pkg>`.

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

Eigen and GTest are fetched automatically if not found. Key options:

| Option | Default | Effect |
|---|---|---|
| `BREW_TARGET` | `desktop` | `desktop` (full library + sampling/plotting) or `hardware` (drops desktop modules, exceptions off, `BREW_ASSERT` instead of throw) |
| `BREW_MODELS` | all 10 packages | Semicolon list of model packages to compile. Lets a C++/hardware build include only what it needs, e.g. `-DBREW_MODELS="gaussian;trajectory_gaussian"` |
| `BREW_ENABLE_PLOTTING` | `ON` | matplot++ plotting (desktop only) |
| `BREW_BUILD_TESTS` | `ON` | GoogleTest suite |

(The MATLAB MEX build always compiles every model regardless of `BREW_MODELS`.)

## Quick Start

```cpp
#include <brew/gaussian/gaussian.hpp>
#include <brew/dynamics/single_integrator.hpp>

using namespace brew;

// Dynamics + single-target filter
auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(2);
auto ekf = std::make_unique<filters::EKF<>>();
ekf->set_dynamics(dyn);
// ... set process/measurement noise ...

// Birth intensity + multi-target filter
auto birth = std::make_unique<models::Mixture<models::Gaussian<>>>();
// ... add birth components ...

multi_target::PHD<models::Gaussian<>> tracker;
tracker.set_filter(std::move(ekf));
tracker.set_birth_model(std::move(birth));
tracker.prob_detection_ = 0.9;
tracker.prob_survive_   = 0.99;

for (const auto& measurements : measurement_sequence) {
    tracker.predict(t, dt);
    tracker.correct(measurements);
    tracker.cleanup();
}
```

Any RFS filter can be swapped in by changing the type (`PHD`, `CPHD`, `MBM`,
`PMBM`, `GLMB`, `JGLMB`); the predict/correct/cleanup interface is the same. Swap
the model package to switch between point and extended targets.

## Testing

```bash
ctest --test-dir build --output-on-failure
```

Per-package, no-MATLAB verification (including the `hardware` target) is available
via `../docker/` — see `docker/README.md`.

## License

See [LICENSE](LICENSE).
