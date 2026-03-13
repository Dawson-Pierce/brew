#pragma once

// Notes: Pure data — no sampling, no pdf, no plotting.

#include "brew/models/base_single_model.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace brew::models {

/// Template-based pose distribution.
/// Holds a translational kinematic state (from dynamics), a rotation matrix on SO(d),
/// and a reference point cloud template. Pure parameter holder.
class TemplatePose : public BaseSingleModel {
public:
    TemplatePose() = default;

    inline TemplatePose(Eigen::VectorXd mean, Eigen::MatrixXd covariance,
                        Eigen::MatrixXd rotation,
                        std::shared_ptr<template_matching::PointCloud> templ,
                        std::vector<int> pos_indices)
        : mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          rotation_(std::move(rotation)),
          template_(std::move(templ)),
          pos_indices_(std::move(pos_indices)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<TemplatePose>(mean_, covariance_, rotation_,
                                              template_, pos_indices_);
    }

    [[nodiscard]] bool is_extended() const override { return true; }

    [[nodiscard]] inline std::unique_ptr<TemplatePose> clone_typed() const {
        return std::make_unique<TemplatePose>(mean_, covariance_, rotation_,
                                              template_, pos_indices_);
    }

    // ---- Kinematic state ----
    [[nodiscard]] const Eigen::VectorXd& mean() const { return mean_; }
    [[nodiscard]] Eigen::VectorXd& mean() { return mean_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return covariance_; }
    [[nodiscard]] Eigen::MatrixXd& covariance() { return covariance_; }

    // ---- Rotation ----
    [[nodiscard]] const Eigen::MatrixXd& rotation() const { return rotation_; }
    [[nodiscard]] Eigen::MatrixXd& rotation() { return rotation_; }

    // ---- Template ----
    [[nodiscard]] const template_matching::PointCloud& get_template() const { return *template_; }
    [[nodiscard]] std::shared_ptr<template_matching::PointCloud> template_ptr() const { return template_; }

    // ---- Position indices ----
    [[nodiscard]] const std::vector<int>& pos_indices() const { return pos_indices_; }

    // ---- Derived queries ----
    [[nodiscard]] int dim() const { return template_ ? template_->dim() : 0; }
    [[nodiscard]] int rot_dim() const { return dim() == 3 ? 3 : 1; }
    [[nodiscard]] int trans_dim() const { return static_cast<int>(mean_.size()); }
    [[nodiscard]] int aug_dim() const { return trans_dim() + rot_dim(); }

    /// Extract position components from mean_ using pos_indices_.
    [[nodiscard]] Eigen::VectorXd position() const {
        const int d = dim();
        Eigen::VectorXd pos(d);
        for (int i = 0; i < d; ++i) {
            pos(i) = mean_(pos_indices_[i]);
        }
        return pos;
    }

    /// Set position components in mean_ using pos_indices_.
    void set_position(const Eigen::VectorXd& pos) {
        const int d = dim();
        for (int i = 0; i < d; ++i) {
            mean_(pos_indices_[i]) = pos(i);
        }
    }

private:
    Eigen::VectorXd mean_;
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd rotation_;
    std::shared_ptr<template_matching::PointCloud> template_;
    std::vector<int> pos_indices_;
};

} // namespace brew::models
