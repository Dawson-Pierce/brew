#pragma once

// Notes: Pure data — no sampling, no plotting.

#include "brew/shared/base_single_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::models {

// @mex model
// @mex_name GGIW
// @mex_fields alpha:scalar, beta:scalar, mean:vec, covariance:mat, v:scalar, V:mat
template<typename T = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class GGIW : public BaseSingleModel {
public:
    using Vector = Eigen::Matrix<T, D, 1>;
    using Matrix = Eigen::Matrix<T, D, D>;
    using ExtentMatrix = Eigen::Matrix<T, De, De>;

    GGIW() = default;

    inline GGIW(T alpha, T beta,
                Vector mean, Matrix covariance,
                T v, ExtentMatrix V)
        : mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          alpha_(alpha), beta_(beta),
          v_(v), V_(std::move(V)) {}

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        return std::make_unique<GGIW<T, D, De>>(alpha_, beta_, mean_, covariance_, v_, V_);
    }

    [[nodiscard]] bool is_extended() const override { return true; }

    [[nodiscard]] inline std::unique_ptr<GGIW<T, D, De>> clone_typed() const {
        return std::make_unique<GGIW<T, D, De>>(alpha_, beta_, mean_, covariance_, v_, V_);
    }

    [[nodiscard]] const Vector& mean() const { return mean_; }
    [[nodiscard]] Vector& mean() { return mean_; }
    [[nodiscard]] const Matrix& covariance() const { return covariance_; }
    [[nodiscard]] Matrix& covariance() { return covariance_; }

    [[nodiscard]] T alpha() const { return alpha_; }
    [[nodiscard]] T& alpha() { return alpha_; }
    [[nodiscard]] T beta() const { return beta_; }
    [[nodiscard]] T& beta() { return beta_; }

    [[nodiscard]] T v() const { return v_; }
    [[nodiscard]] T& v() { return v_; }
    [[nodiscard]] const ExtentMatrix& V() const { return V_; }
    [[nodiscard]] ExtentMatrix& V() { return V_; }
    [[nodiscard]] int extent_dim() const { return static_cast<int>(V_.rows()); }

private:
    Vector mean_;
    Matrix covariance_;
    T alpha_ = T(0);
    T beta_ = T(0);
    T v_ = T(0);
    ExtentMatrix V_;
};

}
