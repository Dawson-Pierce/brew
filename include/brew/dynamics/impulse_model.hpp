#pragma once

#include "brew/models/impulse_prior.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace brew::dynamics {

// A Bernoulli/uniform velocity jump at the start of a prediction interval.
// The supplied transition propagates that jump to the end of the interval.
// For nonlinear motion this is the first-order response about the prior state.
class ImpulseModel {
public:
    struct Prediction {
        Eigen::VectorXd mean_shift, response;
        Eigen::MatrixXd covariance, attitude_jacobian;
    };

    explicit ImpulseModel(const models::ImpulsePrior& prior,
                         Eigen::VectorXd axis = Eigen::Vector3d::UnitX(),
                         std::vector<int> velocity_indices = {3, 4, 5},
                         bool body_frame = false)
        : prior_(prior), axis_(std::move(axis)), indices_(std::move(velocity_indices)),
          body_frame_(body_frame) {
        if (axis_.size() == 0 || !axis_.allFinite() || axis_.norm() == 0.
                || indices_.size() != static_cast<std::size_t>(axis_.size())
                || (body_frame_ && axis_.size() != 3))
            throw std::invalid_argument("Impulse axis and velocity indices must have matching dimensions; body-frame axes must be 3D");
        auto indices = indices_;
        std::sort(indices.begin(), indices.end());
        if (indices.front() < 0 || std::adjacent_find(indices.begin(), indices.end()) != indices.end())
            throw std::invalid_argument("Impulse velocity indices must be distinct and nonnegative");
        axis_.normalize();
    }

    const models::ImpulsePrior& prior() const { return prior_; }
    const Eigen::VectorXd& axis() const { return axis_; }
    const std::vector<int>& velocity_indices() const { return indices_; }
    bool body_frame() const { return body_frame_; }
    double mean() const { return prior_.maximum / 2.0; }
    double variance() const { return prior_.maximum * prior_.maximum / 12.0; }
    double branch_probability(int branch) const {
        return branch == 0 ? 1.0 - prior_.probability : prior_.probability;
    }

    Prediction prediction(const Eigen::MatrixXd& transition,
                          const Eigen::Matrix3d& rotation = Eigen::Matrix3d::Identity(),
                          const Eigen::Matrix3d& attitude_covariance = Eigen::Matrix3d::Zero()) const {
        Eigen::MatrixXd G(transition.rows(), axis_.size());
        for (std::size_t i = 0; i < indices_.size(); ++i) {
            if (indices_[i] >= transition.cols())
                throw std::invalid_argument("Impulse velocity index exceeds the motion state dimension");
            G.col(i) = transition.col(indices_[i]);
        }
        Eigen::VectorXd direction = axis_;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(axis_.size(), 3);
        if (body_frame_) {
            direction = rotation * axis_;
            J << 0.0, direction.z(), -direction.y(),
                -direction.z(), 0.0, direction.x(),
                 direction.y(), -direction.x(), 0.0;
        }
        const Eigen::MatrixXd direction_cov = J * attitude_covariance * J.transpose();
        return {G * (mean() * direction), G * direction,
                variance() * G * (direction * direction.transpose() + direction_cov) * G.transpose(),
                mean() * G * J};
    }

private:
    models::ImpulsePrior prior_;
    Eigen::VectorXd axis_;
    std::vector<int> indices_;
    bool body_frame_;
};

} // namespace brew::dynamics
