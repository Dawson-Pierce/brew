#pragma once

#include "brew/dynamics/nonlinear_dynamics.hpp"
#include <cmath>

namespace brew::dynamics {

// @mex dynamics
// @mex_name DoubleIntegrator3DEuler
/// 3D double integrator with Euler angle state:
/// [x,y,z, vx,vy,vz, ax,ay,az, phi,theta,psi, p,q,r, alpha_x,alpha_y,alpha_z].
template <typename Scalar = double>
class DoubleIntegrator3DEuler : public NonlinearDynamics<Scalar, 18> {
public:
    using Base = NonlinearDynamics<Scalar, 18>;
    using Vector = typename Base::Vector;
    using Matrix = typename Base::Matrix;
    using InputVector = typename Base::InputVector;
    using InputMatrix = typename Base::InputMatrix;

    [[nodiscard]] std::unique_ptr<DynamicsBase<Scalar, 18>> clone() const override {
        return std::make_unique<DoubleIntegrator3DEuler<Scalar>>(*this);
    }

    [[nodiscard]] std::vector<std::string> state_names() const override {
        return {"x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
                "phi", "theta", "psi", "p", "q", "r", "alpha_x", "alpha_y", "alpha_z"};
    }

    [[nodiscard]] Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& input = InputVector{}) const override {

        InputVector u = input.size() > 0 ? input : InputVector::Zero(6);

        Eigen::Matrix<Scalar, 3, 1> pos   = state.template segment<3>(0);
        Eigen::Matrix<Scalar, 3, 1> vel   = state.template segment<3>(3);
        Eigen::Matrix<Scalar, 3, 1> acc   = state.template segment<3>(6);
        Eigen::Matrix<Scalar, 3, 1> eul   = state.template segment<3>(9);
        Eigen::Matrix<Scalar, 3, 1> rates = state.template segment<3>(12);
        Eigen::Matrix<Scalar, 3, 1> alpha = state.template segment<3>(15);

        Eigen::Matrix<Scalar, 3, 1> pos_next = pos + vel * dt + Scalar(0.5) * acc * dt * dt;
        Eigen::Matrix<Scalar, 3, 1> vel_next = vel + acc * dt;
        Eigen::Matrix<Scalar, 3, 1> acc_next = acc;

        const Scalar phi = eul(0);
        const Scalar theta = eul(1);
        Eigen::Matrix<Scalar, 3, 3> T;
        T << Scalar(1), std::sin(phi) * std::tan(theta),  std::cos(phi) * std::tan(theta),
             Scalar(0), std::cos(phi),                   -std::sin(phi),
             Scalar(0), std::sin(phi) / std::cos(theta),  std::cos(phi) / std::cos(theta);

        Eigen::Matrix<Scalar, 3, 1> eul_next = eul + T * rates * dt + Scalar(0.5) * T * alpha * dt * dt;
        Eigen::Matrix<Scalar, 3, 1> rates_next = rates + alpha * dt;
        Eigen::Matrix<Scalar, 3, 1> alpha_next = alpha;

        Vector next;
        next << pos_next, vel_next, acc_next, eul_next, rates_next, alpha_next;

        const InputMatrix G = get_input_mat(dt, state);
        return next + G * u;
    }

    [[nodiscard]] Matrix get_state_mat(
        Scalar dt,
        const Vector& state = Vector{}) const override {

        Matrix F = Matrix::Identity();

        F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
        F(0, 6) = Scalar(0.5) * dt * dt; F(1, 7) = Scalar(0.5) * dt * dt; F(2, 8) = Scalar(0.5) * dt * dt;
        F(3, 6) = dt; F(4, 7) = dt; F(5, 8) = dt;

        const Scalar phi = state(9);
        const Scalar theta = state(10);
        const Scalar p = state(12);
        const Scalar q = state(13);
        const Scalar r = state(14);

        const Scalar cos_phi = std::cos(phi);
        const Scalar sin_phi = std::sin(phi);
        const Scalar cos_theta = std::cos(theta);
        const Scalar tan_theta = std::tan(theta);
        const Scalar sec_theta = Scalar(1) / cos_theta;
        const Scalar sec_theta2 = sec_theta * sec_theta;

        Eigen::Matrix<Scalar, 3, 3> T;
        T << Scalar(1), sin_phi * tan_theta,  cos_phi * tan_theta,
             Scalar(0), cos_phi,             -sin_phi,
             Scalar(0), sin_phi * sec_theta,  cos_phi * sec_theta;

        Eigen::Matrix<Scalar, 3, 3> dT_dphi;
        dT_dphi << Scalar(0), cos_phi * tan_theta, -sin_phi * tan_theta,
                   Scalar(0), -sin_phi,            -cos_phi,
                   Scalar(0), cos_phi * sec_theta, -sin_phi * sec_theta;

        Eigen::Matrix<Scalar, 3, 3> dT_dtheta;
        dT_dtheta << Scalar(0), sin_phi * sec_theta2,  cos_phi * sec_theta2,
                     Scalar(0), Scalar(0),             Scalar(0),
                     Scalar(0), sin_phi * tan_theta * sec_theta,
                                cos_phi * tan_theta * sec_theta;

        Eigen::Matrix<Scalar, 3, 1> rates(p, q, r);
        F.template block<3,1>(9, 9) = dT_dphi * rates * dt;
        F.template block<3,1>(9, 10) = dT_dtheta * rates * dt;
        F.template block<3,3>(9, 12) = T * dt;
        F.template block<3,3>(9, 15) = Scalar(0.5) * T * dt * dt;

        F(12, 15) = dt;
        F(13, 16) = dt;
        F(14, 17) = dt;

        return F;
    }

    [[nodiscard]] InputMatrix get_input_mat(
        Scalar dt,
        const Vector& state = Vector{}) const override {

        InputMatrix G = InputMatrix::Zero(18, 6);

        const Scalar phi = state(9);
        const Scalar theta = state(10);
        Eigen::Matrix<Scalar, 3, 3> T;
        T << Scalar(1), std::sin(phi) * std::tan(theta),  std::cos(phi) * std::tan(theta),
             Scalar(0), std::cos(phi),                   -std::sin(phi),
             Scalar(0), std::sin(phi) / std::cos(theta),  std::cos(phi) / std::cos(theta);

        G.template block<3,3>(0, 0) = Scalar(0.5) * dt * dt * Eigen::Matrix<Scalar, 3, 3>::Identity();
        G.template block<3,3>(3, 0) = dt * Eigen::Matrix<Scalar, 3, 3>::Identity();
        G.template block<3,3>(6, 0) = Eigen::Matrix<Scalar, 3, 3>::Identity();

        G.template block<3,3>(9, 3) = Scalar(0.5) * dt * dt * T;
        G.template block<3,3>(12, 3) = dt * Eigen::Matrix<Scalar, 3, 3>::Identity();
        G.template block<3,3>(15, 3) = Eigen::Matrix<Scalar, 3, 3>::Identity();

        return G;
    }
};

} // namespace brew::dynamics
