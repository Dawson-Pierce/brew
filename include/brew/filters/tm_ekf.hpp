#pragma once

#include "brew/filters/filter.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/template_matching/icp_base.hpp"
#include "brew/template_matching/pca_icp.hpp"
#include <unordered_map>

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

    /// Set the inner ICP algorithm. PCA wrapping is applied automatically
    /// per template on first use — no manual per-template registration needed.
    void set_icp(std::shared_ptr<template_matching::IcpBase> icp) {
        icp_ = std::move(icp);
        icp_map_.clear();  // invalidate cache when base ICP changes
    }

    /// ICP iterations from the last correct() call.
    [[nodiscard]] int last_icp_iterations() const { return last_icp_iterations_; }

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
    mutable int last_icp_iterations_ = 0;
};

} // namespace brew::filters
