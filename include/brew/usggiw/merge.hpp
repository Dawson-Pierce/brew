#pragma once

#include "brew/shared/mixture.hpp"
#include "brew/usggiw/usggiw_model.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace brew::usggiw {

template<typename Scalar, int De, int N>
void merge(models::Mixture<models::USGGIW<Scalar, De>, N>& mix, double threshold) {
    using Model = models::USGGIW<Scalar, De>;
    if (mix.size() < 2) return;
    std::vector<bool> available(mix.size(), true);
    models::Mixture<Model, N> result;
    while (true) {
        std::size_t ref = 0;
        double largest = -1.0;
        bool found = false;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (available[i] && mix.weight(i) > largest) {
                ref = i; largest = mix.weight(i); found = true;
            }
        }
        if (!found) break;
        const auto& reference = mix.component(ref);
        Eigen::Matrix4d Pref = reference.covariance().template cast<double>();
        std::vector<std::size_t> group;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (!available[i]) continue;
            const Eigen::Vector3d component_direction =
                mix.component(i).direction().template cast<double>();
            Eigen::Vector4d dx;
            dx.head<2>() = local_error(reference.direction().template cast<double>(),
                                       component_direction);
            dx.tail<2>() = transport_matrix(
                component_direction, reference.direction().template cast<double>())
                    * mix.component(i).angular_velocity().template cast<double>()
                         - reference.angular_velocity().template cast<double>();
            if (dx.dot(Pref.ldlt().solve(dx)) <= threshold) group.push_back(i);
        }
        double weight = 0.0;
        for (auto i : group) weight += mix.weight(i);
        if (weight <= 0.0) {
            for (auto i : group) available[i] = false;
            continue;
        }
        Eigen::Vector3d resultant = Eigen::Vector3d::Zero();
        for (auto i : group)
            resultant += (mix.weight(i) / weight)
                       * mix.component(i).direction().template cast<double>();
        const Eigen::Vector3d direction = resultant.norm() > 1e-12
            ? resultant.normalized()
            : normalized(reference.direction().template cast<double>());
        Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
        double rate_mean = 0.0, rate_second_moment = 0.0, dof = 0.0;
        Eigen::MatrixXd mean_extent = Eigen::MatrixXd::Zero(
            reference.extent_dim(), reference.extent_dim());
        for (auto i : group) {
            const double q = mix.weight(i) / weight;
            const auto& component = mix.component(i);
            const Eigen::Vector3d component_direction =
                component.direction().template cast<double>();
            const Eigen::Matrix2d T = transport_matrix(component_direction, direction);
            velocity += q * T * component.angular_velocity().template cast<double>();
            const double component_rate_mean = component.alpha() / component.beta();
            rate_mean += q * component_rate_mean;
            rate_second_moment += q * component.alpha() * (component.alpha() + 1.0)
                                / (component.beta() * component.beta());
            dof += q * component.v();
            const double floor = 2.0 * component.extent_dim() + 2.0;
            mean_extent += q * T * component.V().template cast<double>() * T.transpose()
                         / std::max<double>(component.v() - floor, 1e-12);
        }
        const double rate_variance = std::max(
            rate_second_moment - rate_mean * rate_mean, 1e-12);
        const double alpha = rate_mean * rate_mean / rate_variance;
        const double beta = rate_mean / rate_variance;
        const double floor = 2.0 * reference.extent_dim() + 2.0;
        const Eigen::MatrixXd V = std::max(dof - floor, 1e-12) * mean_extent;
        Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();
        for (auto i : group) {
            const double q = mix.weight(i) / weight;
            const Eigen::Vector3d component_direction =
                mix.component(i).direction().template cast<double>();
            const Eigen::Matrix4d G = state_transport_matrix(component_direction, direction);
            Eigen::Vector4d dx;
            dx.head<2>() = local_error(direction, component_direction);
            dx.tail<2>() = transport_matrix(component_direction, direction)
                               * mix.component(i).angular_velocity().template cast<double>()
                         - velocity;
            const Eigen::Matrix4d component_covariance =
                G * mix.component(i).covariance().template cast<double>() * G.transpose();
            covariance += q * (component_covariance + dx * dx.transpose());
            available[i] = false;
        }
        result.add_component(std::make_unique<Model>(
            alpha, beta, direction.template cast<Scalar>(), velocity.template cast<Scalar>(),
            covariance.template cast<Scalar>(), dof, V.template cast<Scalar>()), weight);
    }
    mix = std::move(result);
}

}  // namespace brew::usggiw
