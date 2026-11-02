#pragma once

// Density-specific moment matching used by Mixture::merge.

#include "brew/models/mixture.hpp"
#include "brew/models/gaussian.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <limits>

namespace brew::models {

template <typename Scalar, int D, int N>
void merge(models::Mixture<models::Gaussian<Scalar, D>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double min_dist = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {
                const Eigen::VectorXd diff = mix.component(i).mean() - mix.component(j).mean();
                const Eigen::MatrixXd C = 0.5 * (mix.component(i).covariance() + mix.component(j).covariance());
                const double d2 = diff.transpose() * C.ldlt().solve(diff);
                if (d2 < min_dist) {
                    min_dist = d2;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (min_dist < threshold) {
            const double wi = mix.weight(best_i);
            const double wj = mix.weight(best_j);
            const double w = wi + wj;

            const Eigen::VectorXd& mi = mix.component(best_i).mean();
            const Eigen::VectorXd& mj = mix.component(best_j).mean();
            const Eigen::MatrixXd& Pi = mix.component(best_i).covariance();
            const Eigen::MatrixXd& Pj = mix.component(best_j).covariance();

            Eigen::VectorXd m_new = (wi * mi + wj * mj) / w;
            Eigen::MatrixXd P_new = (wi * (Pi + (mi - m_new) * (mi - m_new).transpose())
                                    + wj * (Pj + (mj - m_new) * (mj - m_new).transpose())) / w;

            mix.component(best_i).mean() = m_new;
            mix.component(best_i).covariance() = P_new;
            mix.merge_augmentation(mix.component(best_i), {best_i, best_j});
            mix.weights()(static_cast<Eigen::Index>(best_i)) = w;
            mix.remove_component(best_j);

            keep_merging = true;
        }
    }
}

}

namespace brew::models {
template<typename Scalar, int D>
struct MixtureMerge<Gaussian<Scalar, D>> {
    static constexpr bool cap_before_merge = false;
    template<int N>
    static void apply(Mixture<Gaussian<Scalar, D>, N>& mixture,
                      double threshold, double rotation_threshold) {
        models::merge(mixture, threshold);
    }
};
}
