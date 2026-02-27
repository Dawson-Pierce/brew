#pragma once

// Ported from: +BREW/+distributions/GGIW.m (basis_tracking logic)
// Ported on: 2026-02-25
// Notes: Extends GGIW with eigenvector basis for orientation tracking.

#include "brew/models/ggiw.hpp"
#include <Eigen/Dense>
#include <memory>

namespace brew::models {

/// GGIW distribution with eigenvector basis tracking for stable orientation.
/// After each correction the IW shape matrix V is decomposed via eigendecomposition;
/// the rotation basis (eigenvectors) and eigenvalues are stored so that subsequent
/// corrections can align the new eigenvectors to the previous ones.
class GGIWOrientation : public GGIW {
public:
    GGIWOrientation() = default;

    inline GGIWOrientation(Eigen::VectorXd mean, Eigen::MatrixXd covariance,
                           double alpha, double beta,
                           double v, Eigen::MatrixXd V)
        : GGIW(std::move(mean), std::move(covariance), alpha, beta, v, std::move(V))
    {
        // Decompose V to initialize basis and eigenvalues
        if (this->V().size() > 0) {
            const int d = extent_dim();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(this->V());
            basis_ = es.eigenvectors();
            eigenvalues_ = es.eigenvalues().asDiagonal();
            has_eigenvalues_ = true;
        }
    }

    [[nodiscard]] inline std::unique_ptr<BaseSingleModel> clone() const override {
        auto c = std::make_unique<GGIWOrientation>(mean(), covariance(), alpha(), beta(), v(), V());
        c->basis_ = basis_;
        c->eigenvalues_ = eigenvalues_;
        c->has_eigenvalues_ = has_eigenvalues_;
        return c;
    }

    [[nodiscard]] inline std::unique_ptr<GGIWOrientation> clone_typed() const {
        auto c = std::make_unique<GGIWOrientation>(mean(), covariance(), alpha(), beta(), v(), V());
        c->basis_ = basis_;
        c->eigenvalues_ = eigenvalues_;
        c->has_eigenvalues_ = has_eigenvalues_;
        return c;
    }

    // ---- Basis tracking accessors ----

    [[nodiscard]] const Eigen::MatrixXd& basis() const { return basis_; }
    [[nodiscard]] Eigen::MatrixXd& basis() { return basis_; }
    void set_basis(const Eigen::MatrixXd& b) { basis_ = b; }

    [[nodiscard]] const Eigen::MatrixXd& eigenvalues() const { return eigenvalues_; }
    [[nodiscard]] Eigen::MatrixXd& eigenvalues() { return eigenvalues_; }
    [[nodiscard]] bool has_eigenvalues() const { return eigenvalues_.size() > 0; }

private:
    Eigen::MatrixXd basis_;          ///< Eigenvector matrix (rotation) from eigendecomposition of V
    Eigen::MatrixXd eigenvalues_;    ///< Diagonal eigenvalue matrix
    bool has_eigenvalues_ = false;
};

} // namespace brew::models
