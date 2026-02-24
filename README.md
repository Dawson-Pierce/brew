# Bayesian Recursive Estimation Workspace (BREW)
---

A header-heavy C++ library for single-target and multi-target tracking, supporting both point and extended targets. Built on Eigen and C++20.

## Modules

| Module | Description |
|---|---|
| `models` | State distribution representations (point, extended, trajectory-aware) |
| `dynamics` | Motion models for prediction |
| `filters` | Single-target recursive filters (prediction + update) |
| `multi_target` | Random Finite Set (RFS) multi-target tracking filters |
| `fusion` | Mixture management operations (prune, merge, cap) |
| `assignment` | Optimal and K-best assignment solvers |
| `clustering` | Measurement clustering for extended targets |
| `metrics` | Performance metrics (OSPA, GOSPA) |
| `serialization` | Filter state serialization (nlohmann/json) |
| `plot_utils` | Optional visualization utilities (requires matplot++) |

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Eigen and GTest are fetched automatically if not found on the system. Plotting support is enabled by default and can be disabled with `-DBREW_ENABLE_PLOTTING=OFF`.

## Quick Start

All multi-target filters share a common template interface parameterized on distribution type. A typical tracking loop looks like:

```cpp
#include <brew/multi_target/pmbm.hpp>
#include <brew/filters/ekf.hpp>
#include <brew/models/gaussian.hpp>
#include <brew/dynamics/integrator_2d.hpp>

// 1. Set up dynamics and single-target filter
auto dynamics = std::make_shared<brew::dynamics::Integrator2D>(dt);
auto ekf = std::make_shared<brew::filters::EKF>(dynamics, R);

// 2. Create birth model
auto birth = std::make_shared<brew::models::Mixture<brew::models::Gaussian>>();
birth->add_component(birth_gaussian, birth_weight);

// 3. Create multi-target filter
brew::multi_target::PMBM<brew::models::Gaussian> tracker(ekf, birth);
tracker.set_prob_detection(0.9);
tracker.set_prob_survive(0.99);
tracker.set_clutter_rate(10.0);

// 4. Predict-update loop
for (const auto& measurements : measurement_sequence) {
    tracker.predict();
    tracker.update(measurements);
    auto estimates = tracker.estimate();
}
```

Any RFS filter can be swapped in by changing the type (e.g. `PHD`, `MBM`, `LMB`, `GLMB`) — the predict/update/estimate interface is the same. Swap the distribution type template parameter to switch between point and extended target tracking.

## Testing

```bash
cd build
ctest --output-on-failure
```

## License

See [LICENSE](LICENSE).
