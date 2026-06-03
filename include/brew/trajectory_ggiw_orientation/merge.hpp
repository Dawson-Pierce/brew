#pragma once

// Component merge for the `trajectory_ggiw_orientation` package -- owns ONLY the TrajectoryGGIWOrientation merge
// (no cross-model overload set). In namespace brew::trajectory_ggiw_orientation so the package's
// concrete RFS resolve an unqualified merge(...) here.
#include "brew/shared/mixture.hpp"
#include "brew/trajectory_ggiw_orientation/trajectory_ggiw_orientation_model.hpp"
#include "brew/shared/fusion/trajectory_mahal.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::trajectory_ggiw_orientation {


template <typename Scalar, int D, int De, int N>
void merge(models::Mixture<models::TrajectoryGGIWOrientation<Scalar, D, De>, N>& mix, double threshold) {
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

            const int len_i = static_cast<int>(mix.component(best_i).mean_history().cols());
            const int len_j = static_cast<int>(mix.component(best_j).mean_history().cols());

            std::size_t keep, drop;
            if (len_i > len_j || (len_i == len_j && wi >= wj)) {
                keep = best_i; drop = best_j;
            } else {
                keep = best_j; drop = best_i;
            }

            auto& base = mix.component(keep);
            const int sd = base.state_dim;
            const int n_total = static_cast<int>(base.mean().size());
            const int start = n_total - sd;

            Eigen::VectorXd m = base.mean();
            Eigen::MatrixXd C = base.covariance();
            m.segment(start, sd) = base.get_last_state();
            C.block(start, start, sd, sd) = base.get_last_cov();
            C = 0.5 * (C + C.transpose());
            base.mean() = m;
            base.covariance() = C;

            mix.weights()(static_cast<Eigen::Index>(keep)) = W;
            mix.remove_component(drop);
            keep_merging = true;
        }
    }
}

}  // namespace brew::trajectory_ggiw_orientation
