#pragma once

// Component merge for the `trajectory_gaussian` package -- owns ONLY the TrajectoryGaussian merge
// (no cross-model overload set). In namespace brew::trajectory_gaussian so the package's
// concrete RFS resolve an unqualified merge(...) here.

#include "brew/shared/mixture.hpp"
#include "brew/trajectory_gaussian/trajectory_gaussian_model.hpp"
#include "brew/shared/fusion/trajectory_mahal.hpp"
#include "brew/shared/fusion/trajectory_moment_merge.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::trajectory_gaussian {

template <typename Scalar, int D, int N>
void merge(models::Mixture<models::TrajectoryGaussian<Scalar, D>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                const double d2 = fusion::trajectory_mahal_dist(
                    mix.component(i).mean(), mix.component(i).covariance(),
                    mix.component(j).mean(), mix.component(j).covariance());
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_d2 < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double W = wi + wj;

            if (!fusion::moment_merge_trajectory(
                    mix.component(best_i), mix.component(best_j), wi, wj)) {
                break;
            }
            mix.weights()(static_cast<Eigen::Index>(best_i)) = W;
            mix.remove_component(best_j);
            keep_merging = true;
        }
    }
}

}
