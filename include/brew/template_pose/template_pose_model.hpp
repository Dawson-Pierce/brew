#pragma once

// Notes: Pure data — no sampling, no pdf, no plotting.

#include "brew/shared/base_single_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace brew::models {

// @mex model
// @mex_name TemplatePose
// @mex_fields mean:vec, covariance:mat, rotation:mat
// @mex_create_int_fields template_id
// @mex_create_int_vec_fields pos_indices
template <typename T = double, int D = Eigen::Dynamic>
class TemplatePose : public BaseSingleModel {
public:
    using Vector = Eigen::Matrix<T, D, 1>;
    using Matrix = Eigen::Matrix<T, D, D>;
    using RotationMatrix = Eigen::Matrix<T, 3, 3>;
    using PositionVector = Eigen::Matrix<T, 3, 1>;

    TemplatePose() = default;

    inline TemplatePose(Vector mean, Matrix covariance,
                        int template_id,
                        std::vector<int> pos_indices,
                        RotationMatrix rotation = RotationMatrix::Identity())
        : mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          rotation_(std::move(rotation)),
          template_id_(template_id),
          pos_indices_(std::move(pos_indices)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return clone_typed();
    }

    [[nodiscard]] bool is_extended() const override { return true; }

    [[nodiscard]] inline std::unique_ptr<TemplatePose<T, D>> clone_typed() const {
        auto out = std::make_unique<TemplatePose<T, D>>();
        out->mean_ = mean_;
        out->covariance_ = covariance_;
        out->rotation_ = rotation_;
        out->needs_pca_alignment_ = needs_pca_alignment_;
        out->template_id_ = template_id_;
        out->pos_indices_ = pos_indices_;
        return out;
    }

    [[nodiscard]] const Vector& mean() const { return mean_; }
    [[nodiscard]] Vector& mean() { return mean_; }
    [[nodiscard]] const Matrix& covariance() const { return covariance_; }
    [[nodiscard]] Matrix& covariance() { return covariance_; }

    [[nodiscard]] const RotationMatrix& rotation() const { return rotation_; }
    [[nodiscard]] RotationMatrix& rotation() { return rotation_; }

    void set_rotation(const RotationMatrix& R) {
        rotation_ = R;
    }

    [[nodiscard]] bool needs_pca_alignment() const { return needs_pca_alignment_; }

    void mark_pca_aligned() { needs_pca_alignment_ = false; }

    [[nodiscard]] TemplatePose with_updated_state(Vector new_mean,
                                                  Matrix new_covariance,
                                                  RotationMatrix new_rotation) const {
        TemplatePose out = *this;
        out.mean_ = std::move(new_mean);
        out.covariance_ = std::move(new_covariance);
        out.rotation_ = std::move(new_rotation);
        return out;
    }

    [[nodiscard]] TemplatePose with_updated_state_aligned(Vector new_mean,
                                                          Matrix new_covariance,
                                                          RotationMatrix new_rotation) const {
        auto out = with_updated_state(std::move(new_mean),
                                      std::move(new_covariance),
                                      std::move(new_rotation));
        out.needs_pca_alignment_ = false;
        return out;
    }

    [[nodiscard]] int template_id() const { return template_id_; }

    [[nodiscard]] const std::vector<int>& pos_indices() const { return pos_indices_; }

    [[nodiscard]] static constexpr int dim() { return 3; }
    [[nodiscard]] static constexpr int rot_dim() { return 3; }
    [[nodiscard]] int trans_dim() const { return static_cast<int>(mean_.size()); }
    [[nodiscard]] int aug_dim() const { return trans_dim() + rot_dim(); }

    [[nodiscard]] PositionVector position() const {
        PositionVector pos;
        for (int i = 0; i < 3; ++i) {
            pos(i) = mean_(pos_indices_[i]);
        }
        return pos;
    }

    void set_position(const PositionVector& pos) {
        for (int i = 0; i < 3; ++i) {
            mean_(pos_indices_[i]) = pos(i);
        }
    }

private:
    Vector mean_;
    Matrix covariance_;
    RotationMatrix rotation_ = RotationMatrix::Identity();
    bool needs_pca_alignment_ = true;
    int template_id_ = -1;
    std::vector<int> pos_indices_;
};

}
