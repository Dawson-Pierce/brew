#pragma once

// Component merge for the `trajectory_iggiw` package -- owns ONLY the TrajectoryIGGIW merge
// (no cross-model overload set). In namespace brew::trajectory_iggiw so the package's
// concrete RFS resolve an unqualified merge(...) here.

#include "brew/shared/mixture.hpp"
#include "brew/trajectory_iggiw/trajectory_iggiw_model.hpp"
#include "brew/shared/fusion/trajectory_mahal.hpp"
#include "brew/shared/fusion/trajectory_moment_merge.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::trajectory_iggiw {

template <typename Scalar, int D, int De, int N>
void merge(models::Mixture<models::TrajectoryIGGIW<Scalar, D, De>, N>& mix, double threshold) {
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

            auto& first = mix.component(best_i);
            const auto& second = mix.component(best_j);
            if (first.window_size() != second.window_size()) break;
            const double qi = wi / W;
            const double qj = wj / W;
            for (int k = 0; k < first.window_size(); ++k) {
                auto& dst = first.history_at(k);
                const auto& src = second.history_at(k);
                const double ai = std::max(dst.alpha(), 2.0 + 1e-9);
                const double aj = std::max(src.alpha(), 2.0 + 1e-9);
                const double intensity_mean = qi * dst.beta() / (ai - 1.0)
                    + qj * src.beta() / (aj - 1.0);
                const double intensity_second =
                    qi * dst.beta() * dst.beta() / ((ai - 1.0) * (ai - 2.0))
                    + qj * src.beta() * src.beta() / ((aj - 1.0) * (aj - 2.0));
                const double intensity_variance = std::max(
                    intensity_second - intensity_mean * intensity_mean, 1e-12);
                dst.alpha() = 2.0 + intensity_mean * intensity_mean / intensity_variance;
                dst.beta() = intensity_mean * (dst.alpha() - 1.0);

                const int de = dst.extent_dim();
                const double dof_floor = 2.0 * de + 2.0;
                const double v_new = qi * dst.v() + qj * src.v();
                const Eigen::MatrixXd mean_extent =
                    qi * dst.V() / std::max(dst.v() - dof_floor, 1e-12)
                    + qj * src.V() / std::max(src.v() - dof_floor, 1e-12);
                dst.v() = v_new;
                dst.V() = std::max(v_new - dof_floor, 1e-12) * mean_extent;
            }
            if (!fusion::moment_merge_trajectory(first, second, wi, wj)) break;
            mix.weights()(static_cast<Eigen::Index>(best_i)) = W;
            mix.remove_component(best_j);
            keep_merging = true;
        }
    }
}

}
