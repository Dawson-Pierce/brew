#pragma once

// Notes: Extends GGIW with eigenvector basis for orientation tracking.

#include "brew/core/models/ggiw.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::models {

/// GGIW distribution with eigenvector basis tracking for stable orientation.
/// After each correction the IW shape matrix V is decomposed via eigendecomposition;
/// the rotation basis (eigenvectors) and eigenvalues are stored so that subsequent
/// corrections can align the new eigenvectors to the previous ones.
// @mex model
// @mex_name GGIWOrientation
// @mex_fields alpha:scalar, beta:scalar, mean:vec, covariance:mat, v:scalar, V:mat
// @mex_extract_extra basis:mat
template<typename T = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class GGIWOrientation : public GGIW<T, D, De> {
public:
    using Base = GGIW<T, D, De>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using ExtentMatrix = typename Base::ExtentMatrix;

    GGIWOrientation() = default;

    inline GGIWOrientation(T alpha, T beta,
                           Vector mean, Matrix covariance,
                           T v, ExtentMatrix V)
        : Base(alpha, beta, std::move(mean), std::move(covariance), v, std::move(V))
    {
        // Decompose V to initialize basis and eigenvalues
        if (this->V().size() > 0) {
            Eigen::SelfAdjointEigenSolver<ExtentMatrix> es(this->V());
            basis_ = es.eigenvectors();
            eigenvalues_ = es.eigenvalues().asDiagonal();
            has_eigenvalues_ = true;
        }
    }

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        auto c = std::make_unique<GGIWOrientation<T, D, De>>(
            this->alpha(), this->beta(), this->mean(), this->covariance(), this->v(), this->V());
        c->basis_ = basis_;
        c->eigenvalues_ = eigenvalues_;
        c->has_eigenvalues_ = has_eigenvalues_;
        return c;
    }

    [[nodiscard]] inline std::unique_ptr<GGIWOrientation<T, D, De>> clone_typed() const {
        auto c = std::make_unique<GGIWOrientation<T, D, De>>(
            this->alpha(), this->beta(), this->mean(), this->covariance(), this->v(), this->V());
        c->basis_ = basis_;
        c->eigenvalues_ = eigenvalues_;
        c->has_eigenvalues_ = has_eigenvalues_;
        return c;
    }

    // ---- Basis tracking accessors ----
    [[nodiscard]] const ExtentMatrix& basis() const { return basis_; }
    [[nodiscard]] ExtentMatrix& basis() { return basis_; }
    void set_basis(const ExtentMatrix& b) { basis_ = b; }

    [[nodiscard]] const ExtentMatrix& eigenvalues() const { return eigenvalues_; }
    [[nodiscard]] ExtentMatrix& eigenvalues() { return eigenvalues_; }
    [[nodiscard]] bool has_eigenvalues() const { return eigenvalues_.size() > 0; }

private:
    ExtentMatrix basis_;          ///< Eigenvector matrix (rotation) from eigendecomposition of V
    ExtentMatrix eigenvalues_;    ///< Diagonal eigenvalue matrix
    bool has_eigenvalues_ = false;
};

} // namespace brew::models
