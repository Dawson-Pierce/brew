#pragma once

#include "brew/shared/mixture.hpp"
#include "brew/trajectory_usggiw/trajectory_usggiw_model.hpp"
#include "brew/usggiw/unit_sphere.hpp"
#include <limits>

namespace brew::trajectory_usggiw {

template <typename Scalar, int De>
double trajectory_distance(const models::TrajectoryUSGGIW<Scalar, De>& left,
                           const models::TrajectoryUSGGIW<Scalar, De>& right) {
    const int overlap = std::min(left.window_size(), right.window_size());
    if (overlap == 0) return std::numeric_limits<double>::infinity();
    double distance = 0.0;
    for (int i = 0; i < overlap; ++i) {
        const int li = left.window_size() - overlap + i;
        const int ri = right.window_size() - overlap + i;
        // Antipodal histories cannot share the local transport chart used for merging.
        const Eigen::Vector3d left_direction = usggiw::normalized(
            left.history_at(li).direction().template cast<double>());
        const Eigen::Vector3d right_direction = usggiw::normalized(
            right.history_at(ri).direction().template cast<double>());
        if (left_direction.dot(right_direction) < -1.0 + 1e-8)
            return std::numeric_limits<double>::infinity();
        Eigen::Vector4d error;
        error.head<2>() = usggiw::local_error(
            left.history_at(li).direction().template cast<double>(),
            right.history_at(ri).direction().template cast<double>());
        const Eigen::Matrix2d velocity_transport = usggiw::transport_matrix(
            right.history_at(ri).direction().template cast<double>(),
            left.history_at(li).direction().template cast<double>());
        error.tail<2>() = velocity_transport
                           * right.history_at(ri).angular_velocity().template cast<double>()
                       - left.history_at(li).angular_velocity().template cast<double>();
        const Eigen::Matrix4d state_transport = usggiw::state_transport_matrix(
            right.history_at(ri).direction().template cast<double>(),
            left.history_at(li).direction().template cast<double>());
        Eigen::Matrix4d covariance = left.cov_at(li, li).template cast<double>()
                                   + state_transport
                                   * right.cov_at(ri, ri).template cast<double>()
                                   * state_transport.transpose();
        covariance = 0.5 * (covariance + covariance.transpose());
        distance += error.dot(covariance.ldlt().solve(error));
    }
    return distance / static_cast<double>(overlap);
}

template <typename Scalar, int De, int N>
void merge(models::Mixture<models::TrajectoryUSGGIW<Scalar, De>, N>& mixture,
           double threshold) {
    while (mixture.size() > 1) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t best_left = 0, best_right = 0;
        for (std::size_t i = 0; i < mixture.size(); ++i) {
            for (std::size_t j = i + 1; j < mixture.size(); ++j) {
                const double distance = trajectory_distance(
                    mixture.component(i), mixture.component(j));
                if (distance < best) {
                    best = distance;
                    best_left = i;
                    best_right = j;
                }
            }
        }
        if (best >= threshold) break;

        const double left_weight = mixture.weight(best_left);
        const double right_weight = mixture.weight(best_right);
        const auto& left = mixture.component(best_left);
        const auto& right = mixture.component(best_right);
        const std::size_t left_history = left.state_history().size();
        const std::size_t right_history = right.state_history().size();
        std::size_t keep = best_left;
        std::size_t drop = best_right;
        if (right_history > left_history
            || (right_history == left_history && right_weight > left_weight)) {
            keep = best_right;
            drop = best_left;
        }
        mixture.weights()(static_cast<Eigen::Index>(keep)) = left_weight + right_weight;
        mixture.remove_component(drop);
    }
}

}  // namespace brew::trajectory_usggiw
