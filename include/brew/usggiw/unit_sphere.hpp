#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "brew/assert.hpp"

namespace brew::usggiw {

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d K;
    K << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return K;
}

inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d& phi) {
    const double a = phi.norm();
    if (a < 1e-12) return Eigen::Matrix3d::Identity() + skew(phi);
    const Eigen::Matrix3d K = skew(phi / a);
    return Eigen::Matrix3d::Identity() + std::sin(a) * K + (1.0 - std::cos(a)) * K * K;
}

inline Eigen::Vector3d normalized(const Eigen::Vector3d& r) {
    const double n = r.norm();
    if (!std::isfinite(n) || n < 1e-12) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        throw std::invalid_argument("sphere vector must be nonzero");
#else
        BREW_ASSERT(false, "sphere vector must be nonzero");
        return Eigen::Vector3d::UnitX();
#endif
    }
    return r / n;
}

inline Eigen::Vector3d east_axis(const Eigen::Vector3d& r) {
    const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d e = z.cross(r);
    if (e.norm() < 1e-8) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        throw std::domain_error("east is undefined at a geographic pole");
#else
        BREW_ASSERT(false, "east is undefined at a geographic pole");
        return Eigen::Vector3d::UnitX();
#endif
    }
    return normalized(e);
}

inline Eigen::Matrix<double, 3, 2> spatial_basis(const Eigen::Vector3d& r) {
    const Eigen::Vector3d direction = normalized(r);
    const Eigen::Vector3d east = east_axis(direction);
    Eigen::Matrix<double, 3, 2> B;
    B.col(0) = east;
    B.col(1) = direction.cross(east);
    return B;
}

inline Eigen::Vector3d apply_position_error(
    const Eigen::Vector3d& r, const Eigen::Vector2d& error) {
    const Eigen::Vector3d direction = normalized(r);
    const Eigen::Vector3d tangent = spatial_basis(direction) * error;
    return normalized(exp_so3(direction.cross(tangent)) * direction);
}

inline Eigen::Vector2d local_error(const Eigen::Vector3d& reference, const Eigen::Vector3d& value) {
    const Eigen::Vector3d r = normalized(reference);
    const Eigen::Vector3d v = normalized(value);
    const double c = std::clamp(r.dot(v), -1.0, 1.0);
    const double a = std::acos(c);
    if (a > M_PI - 1e-8) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        throw std::domain_error("sphere logarithm is undefined at the antipode");
#else
        BREW_ASSERT(false, "sphere logarithm is undefined at the antipode");
        return Eigen::Vector2d::Zero();
#endif
    }
    Eigen::Vector3d tangent = v - c * r;
    if (tangent.norm() > 1e-12) tangent *= a / tangent.norm();
    else tangent.setZero();
    return spatial_basis(r).transpose() * tangent;
}

inline Eigen::Matrix3d parallel_transport_rotation(
    const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
    const Eigen::Vector3d a = normalized(from);
    const Eigen::Vector3d b = normalized(to);
    const double cosine = std::clamp(a.dot(b), -1.0, 1.0);
    if (cosine > 1.0 - 1e-12) return Eigen::Matrix3d::Identity();
    if (cosine < -1.0 + 1e-8) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        throw std::domain_error("parallel transport is undefined between antipodes");
#else
        BREW_ASSERT(false, "parallel transport is undefined between antipodes");
        return Eigen::Matrix3d::Identity();
#endif
    }
    const Eigen::Vector3d axis = normalized(a.cross(b));
    return exp_so3(std::acos(cosine) * axis);
}

inline Eigen::Matrix2d transport_matrix(
    const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
    return spatial_basis(normalized(to)).transpose()
         * parallel_transport_rotation(from, to)
         * spatial_basis(normalized(from));
}

inline Eigen::Matrix4d state_transport_matrix(
    const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
    const Eigen::Matrix2d T = transport_matrix(from, to);
    Eigen::Matrix4d G = Eigen::Matrix4d::Zero();
    G.topLeftCorner<2, 2>() = T;
    G.bottomRightCorner<2, 2>() = T;
    return G;
}

struct PropagatedState {
    Eigen::Vector3d direction;
    Eigen::Vector2d velocity;
    Eigen::Matrix3d rotation;
};

inline Eigen::Matrix3d propagation_rotation(
    const Eigen::Vector3d& direction, const Eigen::Vector2d& velocity, double dt) {
    const Eigen::Vector3d r = normalized(direction);
    const Eigen::Vector3d tangent_velocity = spatial_basis(r) * velocity;
    return exp_so3(dt * r.cross(tangent_velocity));
}

inline PropagatedState propagate(
    const Eigen::Vector3d& direction, const Eigen::Vector2d& velocity, double dt) {
    const Eigen::Vector3d r = normalized(direction);
    const Eigen::Matrix3d rotation = propagation_rotation(direction, velocity, dt);
    const Eigen::Vector3d next_direction = normalized(rotation * r);
    const Eigen::Vector2d next_velocity = spatial_basis(next_direction).transpose()
                                              * rotation * spatial_basis(r) * velocity;
    return {next_direction, next_velocity, rotation};
}

inline Eigen::Matrix2d propagation_extent_transport(
    const Eigen::Vector3d& from, const PropagatedState& propagated) {
    return spatial_basis(propagated.direction).transpose()
         * propagated.rotation * spatial_basis(from);
}

inline Eigen::Matrix4d transition_jacobian(
    const Eigen::Vector3d& direction, const Eigen::Vector2d& velocity, double dt,
    const PropagatedState& nominal) {
    const Eigen::Matrix2d T = propagation_extent_transport(direction, nominal);
    Eigen::Matrix4d F = Eigen::Matrix4d::Zero();
    F.topLeftCorner<2, 2>() = T;
    F.topRightCorner<2, 2>() = dt * T;
    F.bottomRightCorner<2, 2>() = T;
    return F;
}

inline Eigen::Matrix4d transition_jacobian(
    const Eigen::Vector3d& direction, const Eigen::Vector2d& velocity, double dt) {
    return transition_jacobian(direction, velocity, dt, propagate(direction, velocity, dt));
}

}  // namespace brew::usggiw
