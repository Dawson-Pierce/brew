#pragma once

// Notes: Pure data — no sampling, no pdf, no plotting.

#include "brew/shared/base_single_model.hpp"
#include <Eigen/Dense>
#include <limits>
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
                        RotationMatrix rotation = RotationMatrix::Identity(),
                        int mode_id = 0)
        : mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          rotation_(std::move(rotation)),
          template_id_(template_id),
          pos_indices_(std::move(pos_indices)),
          mode_id_(mode_id) {}

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
        out->mode_id_ = mode_id_;
        out->impulse_response_ = impulse_response_;
        out->impulse_observation_ = impulse_observation_;
        out->impulse_variance_ = impulse_variance_;
        out->impulse_branch_ = impulse_branch_;
        out->impulse_parent_ = impulse_parent_;
        out->impulse_log_likelihood_ = impulse_log_likelihood_;
        out->impulse_log_zero_ = impulse_log_zero_;
        out->impulse_log_positive_ = impulse_log_positive_;
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

    // Interval-local latent impulse metadata. Prediction replaces it each scan.
    [[nodiscard]] const Eigen::VectorXd& impulse_response() const { return impulse_response_; }
    void set_impulse_response(Eigen::VectorXd response) { impulse_response_ = std::move(response); }
    [[nodiscard]] double impulse_observation() const { return impulse_observation_; }
    [[nodiscard]] double impulse_variance() const { return impulse_variance_; }
    void set_impulse_evidence(double observation, double variance) {
        impulse_observation_ = observation;
        impulse_variance_ = variance;
    }
    [[nodiscard]] int impulse_branch() const { return impulse_branch_; }
    [[nodiscard]] int impulse_parent() const { return impulse_parent_; }
    void set_impulse_branch(int branch, int parent = -1) {
        impulse_branch_ = branch;
        impulse_parent_ = parent;
    }
    [[nodiscard]] double impulse_log_likelihood() const { return impulse_log_likelihood_; }
    void set_impulse_log_likelihood(double value) { impulse_log_likelihood_ = value; }
    [[nodiscard]] double impulse_log_zero() const { return impulse_log_zero_; }
    [[nodiscard]] double impulse_log_positive() const { return impulse_log_positive_; }
    void set_impulse_branch_evidence(double zero, double positive) {
        impulse_log_zero_ = zero;
        impulse_log_positive_ = positive;
    }

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

    [[nodiscard]] int mode_id() const { return mode_id_; }
    void set_mode_id(int mode_id) { mode_id_ = mode_id; }

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
    int mode_id_ = 0;
    Eigen::VectorXd impulse_response_;
    double impulse_observation_ = std::numeric_limits<double>::quiet_NaN();
    double impulse_variance_ = std::numeric_limits<double>::quiet_NaN();
    int impulse_branch_ = 0;
    int impulse_parent_ = -1;
    double impulse_log_likelihood_ = std::numeric_limits<double>::quiet_NaN();
    double impulse_log_zero_ = std::numeric_limits<double>::quiet_NaN();
    double impulse_log_positive_ = std::numeric_limits<double>::quiet_NaN();
};

}
