#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/template_matching/icp_base.hpp"
#include "brew/template_matching/pca_icp.hpp"
#include <unordered_map>

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

    /// Set the inner ICP algorithm. PCA wrapping is applied automatically
    /// per template on first use.
    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_ = std::move(icp);
        icp_map_.clear();
    }

    void set_window_size(int L) { l_window_ = L; }

private:
    /// Look up the ICP for a given template. For 3D templates, lazily wraps
    /// in PCA-ICP for rotation robustness. For 2D, uses the base ICP directly.
    [[nodiscard]] const template_matching::IcpBase& get_icp(
        const template_matching::PointCloud* templ) const {
        auto it = icp_map_.find(templ);
        if (it != icp_map_.end()) return *it->second;
        if (templ->dim() == 3) {
            auto pca = std::make_shared<template_matching::PcaIcp>(
                icp_->clone(), *templ);
            icp_map_[templ] = pca;
            return *pca;
        }
        return *icp_;
    }

    Eigen::MatrixXd rotation_process_noise_;
    std::shared_ptr<template_matching::IcpBase> icp_;
    mutable std::unordered_map<const template_matching::PointCloud*,
                               std::shared_ptr<template_matching::IcpBase>> icp_map_;
    int l_window_ = 50;
};

} // namespace brew::filters
