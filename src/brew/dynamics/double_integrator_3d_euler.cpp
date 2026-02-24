// Ported from: +BREW/+dynamics/DoubleIntegrator_3D_euler.m
// Original name: DoubleIntegrator_3D_euler
// Ported on: 2026-02-07
// Notes: Euler-angle dynamics; matches MATLAB Jacobian structure.

#include "brew/dynamics/double_integrator_3d_euler.hpp"
#include <cmath>

namespace brew::dynamics {

std::unique_ptr<DynamicsBase> DoubleIntegrator3DEuler::clone() const {
    return std::make_unique<DoubleIntegrator3DEuler>(*this);
}

std::vector<std::string> DoubleIntegrator3DEuler::state_names() const {
    return {"x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
            "phi", "theta", "psi", "p", "q", "r", "alpha_x", "alpha_y", "alpha_z"};
}

Eigen::VectorXd DoubleIntegrator3DEuler::propagate_state(
    double dt,
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& input) const {

    Eigen::VectorXd u = input.size() > 0 ? input : Eigen::VectorXd::Zero(6);

    Eigen::Vector3d pos = state.segment<3>(0);
    Eigen::Vector3d vel = state.segment<3>(3);
    Eigen::Vector3d acc = state.segment<3>(6);
    Eigen::Vector3d eul = state.segment<3>(9);
    Eigen::Vector3d rates = state.segment<3>(12);
    Eigen::Vector3d alpha = state.segment<3>(15);

    Eigen::Vector3d pos_next = pos + vel * dt + 0.5 * acc * dt * dt;
    Eigen::Vector3d vel_next = vel + acc * dt;
    Eigen::Vector3d acc_next = acc;

    const double phi = eul(0);
    const double theta = eul(1);
    Eigen::Matrix3d T;
    T << 1.0, std::sin(phi) * std::tan(theta),  std::cos(phi) * std::tan(theta),
         0.0, std::cos(phi),                   -std::sin(phi),
         0.0, std::sin(phi) / std::cos(theta),  std::cos(phi) / std::cos(theta);

    Eigen::Vector3d eul_next = eul + T * rates * dt + 0.5 * T * alpha * dt * dt;
    Eigen::Vector3d rates_next = rates + alpha * dt;
    Eigen::Vector3d alpha_next = alpha;

    Eigen::VectorXd next(18);
    next << pos_next, vel_next, acc_next, eul_next, rates_next, alpha_next;

    const Eigen::MatrixXd G = get_input_mat(dt, state);
    return next + G * u;
}

Eigen::MatrixXd DoubleIntegrator3DEuler::get_state_mat(
    double dt,
    const Eigen::VectorXd& state) const {

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(18, 18);

    F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
    F(0, 6) = 0.5 * dt * dt; F(1, 7) = 0.5 * dt * dt; F(2, 8) = 0.5 * dt * dt;
    F(3, 6) = dt; F(4, 7) = dt; F(5, 8) = dt;

    const double phi = state(9);
    const double theta = state(10);
    const double p = state(12);
    const double q = state(13);
    const double r = state(14);

    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    const double cos_theta = std::cos(theta);
    const double tan_theta = std::tan(theta);
    const double sec_theta = 1.0 / cos_theta;
    const double sec_theta2 = sec_theta * sec_theta;

    Eigen::Matrix3d T;
    T << 1.0, sin_phi * tan_theta,  cos_phi * tan_theta,
         0.0, cos_phi,             -sin_phi,
         0.0, sin_phi * sec_theta,  cos_phi * sec_theta;

    Eigen::Matrix3d dT_dphi;
    dT_dphi << 0.0, cos_phi * tan_theta, -sin_phi * tan_theta,
               0.0, -sin_phi,            -cos_phi,
               0.0, cos_phi * sec_theta, -sin_phi * sec_theta;

    Eigen::Matrix3d dT_dtheta;
    dT_dtheta << 0.0, sin_phi * sec_theta2,  cos_phi * sec_theta2,
                 0.0, 0.0,                  0.0,
                 0.0, sin_phi * tan_theta * sec_theta,
                      cos_phi * tan_theta * sec_theta;

    Eigen::Vector3d rates(p, q, r);
    F.block<3,1>(9, 9) = dT_dphi * rates * dt;
    F.block<3,1>(9, 10) = dT_dtheta * rates * dt;
    F.block<3,3>(9, 12) = T * dt;
    F.block<3,3>(9, 15) = 0.5 * T * dt * dt;

    F(12, 15) = dt;
    F(13, 16) = dt;
    F(14, 17) = dt;

    return F;
}

Eigen::MatrixXd DoubleIntegrator3DEuler::get_input_mat(
    double dt,
    const Eigen::VectorXd& state) const {

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(18, 6);

    const double phi = state(9);
    const double theta = state(10);
    Eigen::Matrix3d T;
    T << 1.0, std::sin(phi) * std::tan(theta),  std::cos(phi) * std::tan(theta),
         0.0, std::cos(phi),                   -std::sin(phi),
         0.0, std::sin(phi) / std::cos(theta),  std::cos(phi) / std::cos(theta);

    G.block<3,3>(0, 0) = 0.5 * dt * dt * Eigen::Matrix3d::Identity();
    G.block<3,3>(3, 0) = dt * Eigen::Matrix3d::Identity();
    G.block<3,3>(6, 0) = Eigen::Matrix3d::Identity();

    G.block<3,3>(9, 3) = 0.5 * dt * dt * T;
    G.block<3,3>(12, 3) = dt * Eigen::Matrix3d::Identity();
    G.block<3,3>(15, 3) = Eigen::Matrix3d::Identity();

    return G;
}

} // namespace brew::dynamics
