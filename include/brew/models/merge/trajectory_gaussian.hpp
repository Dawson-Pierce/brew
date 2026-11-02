#pragma once

// Density-specific moment matching used by Mixture::merge.

#include "brew/models/mixture.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/utils/fusion/trajectory_mahal.hpp"
#include "brew/utils/fusion/trajectory_moment_merge.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::models {

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
                const double d2 = utils::fusion::trajectory_mahal_dist(
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

            if (!utils::fusion::moment_merge_trajectory(
                    mix.component(best_i), mix.component(best_j), wi, wj)) {
                break;
            }
            mix.merge_augmentation(mix.component(best_i), {best_i, best_j});
            mix.weights()(static_cast<Eigen::Index>(best_i)) = W;
            mix.remove_component(best_j);
            keep_merging = true;
        }
    }
}

}

namespace brew::models {
template<typename Scalar, int D>
struct MixtureMerge<TrajectoryGaussian<Scalar, D>> {
    static constexpr bool cap_before_merge = false;
    template<int N>
    static void apply(Mixture<TrajectoryGaussian<Scalar, D>, N>& mixture,
                      double threshold, double rotation_threshold) {
        models::merge(mixture, threshold);
    }
};
}
