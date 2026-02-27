#pragma once

// Extends TrajectoryGGIW with eigenvector basis tracking for stable orientation.
// Combines windowed trajectory history with SVD-based basis alignment.

#include "brew/models/trajectory_base_model.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace brew::models {

/// GGIW trajectory distribution with eigenvector basis tracking.
/// Combines windowed trajectory history (stacked states, covariance history,
/// alpha/beta/v/V histories) with eigenvector basis and eigenvalues for
/// stable orientation tracking of extended targets.
class TrajectoryGGIWOrientation : public TrajectoryBaseModel {
public:
    TrajectoryGGIWOrientation() = default;

    inline TrajectoryGGIWOrientation(int idx, int state_dim,
                                     Eigen::VectorXd mean, Eigen::MatrixXd covariance,
                                     double alpha, double beta,
                                     double v, Eigen::MatrixXd V)
        : TrajectoryBaseModel(idx, state_dim),
          mean_(std::move(mean)),
          covariance_(std::move(covariance)),
          alpha_(alpha), beta_(beta),
          v_(v), V_(std::move(V))
    {
        const int steps = (state_dim > 0) ? static_cast<int>(mean_.size() / state_dim) : 0;
        window_size = std::max(1, steps);
        if (state_dim > 0 && mean_.size() > 0 && steps > 0) {
            mean_history_ = Eigen::Map<const Eigen::MatrixXd>(
                mean_.data(), state_dim, steps);
        } else {
            mean_history_.resize(0, 0);
        }
        cov_history_.clear();
        cov_history_.push_back(covariance_);
        alpha_history_.assign(1, alpha_);
        beta_history_.assign(1, beta_);
        v_history_.assign(1, v_);
        V_history_.assign(1, V_);

        // Decompose V to initialize basis and eigenvalues
        if (V_.size() > 0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V_);
            basis_ = es.eigenvectors();
            eigenvalues_ = es.eigenvalues().asDiagonal();
            has_eigenvalues_ = true;
        }
    }

    [[nodiscard]] bool is_extended() const override { return true; }

    [[nodiscard]] inline std::unique_ptr<TrajectoryGGIWOrientation> clone() const {
        auto c = std::make_unique<TrajectoryGGIWOrientation>(
            init_idx, state_dim, mean_, covariance_, alpha_, beta_, v_, V_);
        c->window_size = window_size;
        c->basis_ = basis_;
        c->eigenvalues_ = eigenvalues_;
        c->has_eigenvalues_ = has_eigenvalues_;
        return c;
    }

    [[nodiscard]] std::unique_ptr<TrajectoryGGIWOrientation> clone_typed() const { return clone(); }

    // ---- Kinematic state ----
    [[nodiscard]] const Eigen::VectorXd& mean() const { return mean_; }
    [[nodiscard]] Eigen::VectorXd& mean() { return mean_; }
    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return covariance_; }
    [[nodiscard]] Eigen::MatrixXd& covariance() { return covariance_; }
    [[nodiscard]] const Eigen::MatrixXd& mean_history() const { return mean_history_; }
    [[nodiscard]] Eigen::MatrixXd& mean_history() { return mean_history_; }
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& cov_history() const { return cov_history_; }
    [[nodiscard]] std::vector<Eigen::MatrixXd>& cov_history() { return cov_history_; }

    // ---- Gamma ----
    [[nodiscard]] double alpha() const { return alpha_; }
    [[nodiscard]] double& alpha() { return alpha_; }
    [[nodiscard]] double beta() const { return beta_; }
    [[nodiscard]] double& beta() { return beta_; }
    [[nodiscard]] const std::vector<double>& alpha_history() const { return alpha_history_; }
    [[nodiscard]] std::vector<double>& alpha_history() { return alpha_history_; }
    [[nodiscard]] const std::vector<double>& beta_history() const { return beta_history_; }
    [[nodiscard]] std::vector<double>& beta_history() { return beta_history_; }

    // ---- Inverse-Wishart ----
    [[nodiscard]] double v() const { return v_; }
    [[nodiscard]] double& v() { return v_; }
    [[nodiscard]] const Eigen::MatrixXd& V() const { return V_; }
    [[nodiscard]] Eigen::MatrixXd& V() { return V_; }
    [[nodiscard]] const std::vector<double>& v_history() const { return v_history_; }
    [[nodiscard]] std::vector<double>& v_history() { return v_history_; }
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& V_history() const { return V_history_; }
    [[nodiscard]] std::vector<Eigen::MatrixXd>& V_history() { return V_history_; }
    [[nodiscard]] int extent_dim() const { return static_cast<int>(V_.rows()); }

    // ---- Basis tracking ----
    [[nodiscard]] const Eigen::MatrixXd& basis() const { return basis_; }
    [[nodiscard]] Eigen::MatrixXd& basis() { return basis_; }
    void set_basis(const Eigen::MatrixXd& b) { basis_ = b; }
    [[nodiscard]] const Eigen::MatrixXd& eigenvalues() const { return eigenvalues_; }
    [[nodiscard]] Eigen::MatrixXd& eigenvalues() { return eigenvalues_; }
    [[nodiscard]] bool has_eigenvalues() const { return eigenvalues_.size() > 0; }

    // ---- Trajectory helpers ----
    [[nodiscard]] inline Eigen::VectorXd get_last_state() const {
        return mean_.tail(state_dim);
    }

    [[nodiscard]] inline Eigen::MatrixXd get_last_cov() const {
        const int n = state_dim;
        const int total = static_cast<int>(mean_.size());
        return covariance_.block(total - n, total - n, n, n);
    }

    [[nodiscard]] inline Eigen::MatrixXd rearrange_states() const {
        if (state_dim <= 0) return Eigen::MatrixXd();
        const int steps = static_cast<int>(mean_.size() / state_dim);
        return Eigen::Map<const Eigen::MatrixXd>(mean_.data(), state_dim, steps);
    }

private:
    Eigen::VectorXd mean_;
    Eigen::MatrixXd covariance_;
    double alpha_ = 0.0;
    double beta_ = 0.0;
    double v_ = 0.0;
    Eigen::MatrixXd V_;
    Eigen::MatrixXd mean_history_;
    std::vector<Eigen::MatrixXd> cov_history_;
    std::vector<double> alpha_history_;
    std::vector<double> beta_history_;
    std::vector<double> v_history_;
    std::vector<Eigen::MatrixXd> V_history_;

    Eigen::MatrixXd basis_;          ///< Eigenvector matrix (rotation) from eigendecomposition of V
    Eigen::MatrixXd eigenvalues_;    ///< Diagonal eigenvalue matrix
    bool has_eigenvalues_ = false;
};

} // namespace brew::models
