#pragma once

// SO(3) logarithm and exponential maps. Used by template-matching filters
// for rotation innovation and update.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace brew::template_matching::so3 {

[[nodiscard]] inline Eigen::Vector3d log(const Eigen::Matrix3d& R) {
    const double cos_angle = std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);

    if (angle < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    if (angle > M_PI - 1e-6) {

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

// Weighted geodesic (Karcher) mean of rotations on SO(3).
[[nodiscard]] inline Eigen::Matrix3d weighted_mean(const std::vector<Eigen::Matrix3d>& R,
                                                   const std::vector<double>& w) {
    Eigen::Matrix3d Rf = R[0];
    for (int it = 0; it < 20; ++it) {
        Eigen::Vector3d delta = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < R.size(); ++i) delta += w[i] * log(Rf.transpose() * R[i]);
        Rf = Rf * exp(delta);
        if (delta.norm() < 1e-12) break;
    }
    return Rf;
}

}
