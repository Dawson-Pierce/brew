#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/template_matching/icp_base.hpp"

namespace brew::filters {

/// Trajectory EKF for template-matching pose tracking.
/// Stacks translational kinematic state across time (like TrajectoryGaussianEKF),
/// while rotation lives on SO(d) and is tracked only on the current step.

class TrajectoryTmEkf : public Filter<models::Trajectory<models::TemplatePose>> {
public:
    using TrajectoryType = models::Trajectory<models::TemplatePose>;

    TrajectoryTmEkf() = default;

    [[nodiscard]] std::unique_ptr<Filter<TrajectoryType>> clone() const override;

    [[nodiscard]] TrajectoryType predict(
        double dt,
        const TrajectoryType& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const TrajectoryType& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const TrajectoryType& predicted) const override;

    void set_rotation_process_noise(Eigen::MatrixXd Q_rot) {
        rotation_process_noise_ = std::move(Q_rot);
    }

    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_ = std::move(icp);
    }

    void set_window_size(int L) { l_window_ = L; }

private:
    Eigen::MatrixXd rotation_process_noise_;
    std::shared_ptr<template_matching::IcpBase> icp_;
    int l_window_ = 50;
};

} // namespace brew::filters
