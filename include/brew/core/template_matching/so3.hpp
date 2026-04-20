#pragma once

// SO(3) logarithm and exponential maps. Used by template-matching filters
// for rotation innovation and update.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace brew::template_matching::so3 {

/// Log map: SO(3) → so(3). Returns the axis-angle 3-vector whose norm is the
/// rotation angle. Handles the near-identity and near-π edge cases.
[[nodiscard]] inline Eigen::Vector3d log(const Eigen::Matrix3d& R) {
    const double cos_angle = std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);

    if (angle < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    if (angle > M_PI - 1e-6) {
        // Near π: sin(angle) ≈ 0, so the skew-symmetric extraction fails.
        // Recover the rotation axis from R + I (eigenvector with eigenvalue +1).
        Eigen::Matrix3d M = R + Eigen::Matrix3d::Identity();
        Eigen::Vector3d axis = M.col(0);
        for (int i = 1; i < 3; ++i) {
            if (M.col(i).squaredNorm() > axis.squaredNorm()) axis = M.col(i);
        }
        axis.normalize();
        return axis * M_PI;
    }

    Eigen::Vector3d axis;
    axis << R(2, 1) - R(1, 2),
            R(0, 2) - R(2, 0),
            R(1, 0) - R(0, 1);
    axis *= angle / (2.0 * std::sin(angle));
    return axis;
}

/// Exp map: so(3) → SO(3). Rodrigues formula.
[[nodiscard]] inline Eigen::Matrix3d exp(const Eigen::Vector3d& phi) {
    const double angle = phi.norm();
    if (angle < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d axis = phi / angle;
    Eigen::Matrix3d K;
    K <<     0.0, -axis(2),  axis(1),
         axis(2),      0.0, -axis(0),
        -axis(1),  axis(0),      0.0;

    return Eigen::Matrix3d::Identity()
           + std::sin(angle) * K
           + (1.0 - std::cos(angle)) * K * K;
}

} // namespace brew::template_matching::so3
