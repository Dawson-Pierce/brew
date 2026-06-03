#pragma once

// Shared full-state Mahalanobis distance for trajectory merges.
//
// Trajectory models (TrajectoryGaussian/TrajectoryGGIW/...) live in separate
// packages but all merge their components by the SAME full-state Mahalanobis
// gate, so the helper lives here (inline, ODR-safe) and is included by each
// trajectory package's merge.hpp. It is model-agnostic: it only sees stacked
// state means and covariances.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

namespace brew::fusion {

inline double trajectory_mahal_dist(const Eigen::VectorXd& mi, const Eigen::MatrixXd& Pi,
                                    const Eigen::VectorXd& mj, const Eigen::MatrixXd& Pj) {
    if (mi.size() != mj.size()) return std::numeric_limits<double>::infinity();

    Eigen::MatrixXd C = 0.5 * (Pi + Pj);
    C = 0.5 * (C + C.transpose());

    Eigen::LLT<Eigen::MatrixXd> llt(C);
    if (llt.info() != Eigen::Success) {
        const double eps_reg = 1e-9 * std::max(1.0, C.trace() / C.rows());
        C += eps_reg * Eigen::MatrixXd::Identity(C.rows(), C.cols());
        llt.compute(C);
        if (llt.info() != Eigen::Success) {

            const Eigen::VectorXd diff = mi - mj;
            return diff.transpose() * C.completeOrthogonalDecomposition().pseudoInverse() * diff;
        }
    }
    const Eigen::VectorXd y = llt.matrixL().solve(mi - mj);
    return y.squaredNorm();
}

}
