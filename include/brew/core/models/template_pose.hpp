#pragma once

// Notes: Pure data — no sampling, no pdf, no plotting.

#include "brew/core/models/base_single_model.hpp"
#include "brew/core/template_matching/point_cloud.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace brew::models {

/// Template-based pose distribution.
/// Holds a translational kinematic state (from dynamics), a rotation matrix on SO(d),
/// and a reference point cloud template. Pure parameter holder.
// @mex model
// @mex_name TemplatePose
// @mex_fields mean:vec, covariance:mat, rotation:mat
// @mex_create_mat_fields template_points:PointCloud
// @mex_create_int_vec_fields pos_indices
template <typename T = double, int D = Eigen::Dynamic>
class TemplatePose : public BaseSingleModel {
public:
    using Vector = Eigen::Matrix<T, D, 1>;
    using Matrix = Eigen::Matrix<T, D, D>;
    using RotationMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using PositionVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    TemplatePose() = default;

    inline TemplatePose(Vector mean, Matrix covariance,
                        RotationMatrix rotation,
                        std::shared_ptr<template_matching::PointCloud> templ,
                        std::vector<int> pos_indices)
        : mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          rotation_(std::move(rotation)),
          template_(std::move(templ)),
          pos_indices_(std::move(pos_indices)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<TemplatePose<T, D>>(mean_, covariance_, rotation_,
                                                    template_, pos_indices_);
    }

    [[nodiscard]] bool is_extended() const override { return true; }

    [[nodiscard]] inline std::unique_ptr<TemplatePose<T, D>> clone_typed() const {
        return std::make_unique<TemplatePose<T, D>>(mean_, covariance_, rotation_,
                                                    template_, pos_indices_);
    }

    // ---- Kinematic state ----
    [[nodiscard]] const Vector& mean() const { return mean_; }
    [[nodiscard]] Vector& mean() { return mean_; }
    [[nodiscard]] const Matrix& covariance() const { return covariance_; }
    [[nodiscard]] Matrix& covariance() { return covariance_; }

    // ---- Rotation ----
    [[nodiscard]] const RotationMatrix& rotation() const { return rotation_; }
    [[nodiscard]] RotationMatrix& rotation() { return rotation_; }
    [[nodiscard]] bool has_rotation() const { return rotation_.size() > 0; }

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
    [[nodiscard]] PositionVector position() const {
        const int d = dim();
        PositionVector pos(d);
        for (int i = 0; i < d; ++i) {
            pos(i) = mean_(pos_indices_[i]);
        }
        return pos;
    }

    /// Set position components in mean_ using pos_indices_.
    void set_position(const PositionVector& pos) {
        const int d = dim();
        for (int i = 0; i < d; ++i) {
            mean_(pos_indices_[i]) = pos(i);
        }
    }

private:
    Vector mean_;
    Matrix covariance_;
    RotationMatrix rotation_;
    std::shared_ptr<template_matching::PointCloud> template_;
    std::vector<int> pos_indices_;
};

} // namespace brew::models
