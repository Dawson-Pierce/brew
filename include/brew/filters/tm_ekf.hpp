#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/template_matching/icp_base.hpp"

namespace brew::filters {

/// Template Matching Extended Kalman Filter.
/// Invariant EKF for pose estimation using ICP-based template matching.

class TmEkf : public Filter<models::TemplatePose> {
public:
    TmEkf() = default;

    [[nodiscard]] std::unique_ptr<Filter<models::TemplatePose>> clone() const override;

    [[nodiscard]] models::TemplatePose predict(
        double dt,
        const models::TemplatePose& prev) const override;

    [[nodiscard]] CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const models::TemplatePose& predicted) const override;

    [[nodiscard]] double gate(
        const Eigen::VectorXd& measurement,
        const models::TemplatePose& predicted) const override;

    // ---- TM-EKF-specific configuration ----
    void set_rotation_process_noise(Eigen::MatrixXd Q_rot) {
        rotation_process_noise_ = std::move(Q_rot);
    }

    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_ = std::move(icp);
    }

    /// ICP iterations from the last correct() call.
    [[nodiscard]] int last_icp_iterations() const { return last_icp_iterations_; }

private:
    Eigen::MatrixXd rotation_process_noise_;
    std::shared_ptr<template_matching::IcpBase> icp_;
    mutable int last_icp_iterations_ = 0;
};

} // namespace brew::filters
