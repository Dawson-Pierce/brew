#pragma once

// Component merge for the `trajectory_template_pose` package -- owns ONLY the TrajectoryTemplatePose merge
// (no cross-model overload set). In namespace brew::trajectory_template_pose so the package's
// concrete RFS resolve an unqualified merge(...) here.

#include "brew/shared/mixture.hpp"
#include "brew/trajectory_template_pose/trajectory_template_pose_model.hpp"
#include "brew/shared/fusion/trajectory_mahal.hpp"
#include "brew/template_matching/so3.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::trajectory_template_pose {

template <typename Scalar, int D, int N>
void merge(models::Mixture<models::TrajectoryTemplatePose<Scalar, D>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    bool keep_merging = true;
    while (keep_merging && mix.size() > 1) {
        keep_merging = false;

        double best_d2 = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_j = 0;

        for (std::size_t i = 0; i < mix.size(); ++i) {
            for (std::size_t j = i + 1; j < mix.size(); ++j) {

                if (mix.component(i).current().template_id() !=
                    mix.component(j).current().template_id()) continue;
                if (mix.component(i).current().mode_id() !=
                    mix.component(j).current().mode_id()) continue;

                if (!mix.component(i).current().needs_pca_alignment()
                    && !mix.component(j).current().needs_pca_alignment()) {
                    const auto& Ri = mix.component(i).current().rotation();
                    const auto& Rj = mix.component(j).current().rotation();
                    Eigen::MatrixXd R_err = Ri.transpose() * Rj;
                    double cos_a = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
                    if (std::acos(cos_a) > M_PI / 2.0) continue;
                }

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
            const int live = first.stacked_size();
            const double qi = wi / W;
            const double qj = wj / W;
            const Eigen::VectorXd mi = first.mean().head(live);
            const Eigen::VectorXd mj = second.mean().head(live);
            const Eigen::VectorXd merged_mean = qi * mi + qj * mj;
            const Eigen::VectorXd di = mi - merged_mean;
            const Eigen::VectorXd dj = mj - merged_mean;
            Eigen::MatrixXd merged_cov =
                qi * (first.covariance().topLeftCorner(live, live) + di * di.transpose())
                + qj * (second.covariance().topLeftCorner(live, live) + dj * dj.transpose());
            merged_cov = 0.5 * (merged_cov + merged_cov.transpose());

            for (int k = 0; k < first.window_size(); ++k) {
                const auto first_state = first.history_at(k);
                const auto& second_state = second.history_at(k);
                Eigen::Matrix3d rotation_sum =
                    qi * first_state.rotation() + qj * second_state.rotation();
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                    rotation_sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d U = svd.matrixU();
                Eigen::Matrix3d V = svd.matrixV();
                Eigen::Matrix3d sign = Eigen::Matrix3d::Identity();
                sign(2, 2) = (U * V.transpose()).determinant();
                const Eigen::Matrix3d merged_rotation = U * sign * V.transpose();

                const int sd = first.state_dim;
                const int aug = first_state.aug_dim();
                Eigen::VectorXd offset_i = Eigen::VectorXd::Zero(aug);
                Eigen::VectorXd offset_j = Eigen::VectorXd::Zero(aug);
                offset_i.head(sd) = first_state.mean() - merged_mean.segment(k * sd, sd);
                offset_j.head(sd) = second_state.mean() - merged_mean.segment(k * sd, sd);
                offset_i.tail(3) = template_matching::so3::log(
                    first_state.rotation() * merged_rotation.transpose());
                offset_j.tail(3) = template_matching::so3::log(
                    second_state.rotation() * merged_rotation.transpose());
                Eigen::MatrixXd aug_cov =
                    qi * (first_state.covariance() + offset_i * offset_i.transpose())
                    + qj * (second_state.covariance() + offset_j * offset_j.transpose());
                aug_cov = 0.5 * (aug_cov + aug_cov.transpose());
                const auto& base_state = first_state.needs_pca_alignment()
                    && !second_state.needs_pca_alignment() ? second_state : first_state;
                first.history_at(k) = base_state.with_updated_state(
                    merged_mean.segment(k * sd, sd), aug_cov, merged_rotation);
            }

            first.mean().head(live) = merged_mean;
            first.covariance().topLeftCorner(live, live) = merged_cov;
            mix.weights()(static_cast<Eigen::Index>(best_i)) = W;
            mix.remove_component(best_j);
            keep_merging = true;
        }
    }
}

}
