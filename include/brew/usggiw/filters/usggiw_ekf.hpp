#pragma once

#include "brew/shared/filter_base.hpp"
#include "brew/shared/filter_traits.hpp"
#include "brew/shared/parameter_validation.hpp"
#include "brew/usggiw/usggiw_model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace brew::filters {

// @mex filter
// @mex_name USGGIWEKF
// @mex_dist USGGIW
// @mex_setters temporal_decay:scalar, forgetting_factor:scalar, scaling_parameter:scalar, ci_weight:scalar
template<typename Scalar = double, int De = Eigen::Dynamic>
class USGGIWEKF : public Filter<models::USGGIW<Scalar, De>> {
public:
    using Dist = models::USGGIW<Scalar, De>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto out = std::make_unique<USGGIWEKF>();
        out->process_noise_ = this->process_noise_;
        out->measurement_noise_ = this->measurement_noise_;
        out->eta_ = eta_; out->tau_ = tau_; out->rho_ = rho_;
        out->ci_weight_ = ci_weight_;
        return out;
    }

    [[nodiscard]] Dist predict(double dt, const Dist& prev) const override {
        const Eigen::Vector3d r = prev.direction().template cast<double>();
        const Eigen::Vector2d w = prev.angular_velocity().template cast<double>();
        const usggiw::PropagatedState next = usggiw::propagate(r, w, dt);
        const Eigen::Matrix4d F = usggiw::transition_jacobian(r, w, dt, next);
        Eigen::Matrix4d P = F * prev.covariance().template cast<double>() * F.transpose();
        if (this->process_noise_.rows() == 4) P += this->process_noise_;
        P = 0.5 * (P + P.transpose());
        const int d = prev.extent_dim();
        const double floor = 2.0 * d + 2.0;
        const double nv = floor + std::exp(-dt / tau_) * (prev.v() - floor);
        const double scale = (nv - floor) / std::max<double>(prev.v() - floor, 1e-12);
        const Eigen::Matrix2d T = usggiw::propagation_extent_transport(r, next);
        const Eigen::Matrix2d V = scale * T * prev.V().template cast<double>() * T.transpose();
        return Dist(prev.alpha() / eta_, prev.beta() / eta_, next.direction.template cast<Scalar>(),
                    next.velocity.template cast<Scalar>(), P.template cast<Scalar>(), nv,
                    V.template cast<Scalar>());
    }

    [[nodiscard]] CorrectionResult correct(const typename Base::MeasVector& measurement,
                                            const Dist& predicted) const override {
        constexpr int ambient_dim = 3;
        constexpr int extent_dim = 2;
        if (measurement.size() == 0 || measurement.size() % ambient_dim != 0)
            throw std::invalid_argument("USGGIW measurements must be a nonempty 3xN set");
        const int count = static_cast<int>(measurement.size()) / ambient_dim;
        Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> Z(measurement.data(), 3, count);
        const Eigen::Vector3d reference = predicted.direction().template cast<double>();
        Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, count);
        for (int j = 0; j < count; ++j)
            Y.col(j) = usggiw::local_error(reference, Z.col(j));
        const Eigen::Vector2d innovation = Y.rowwise().mean();
        Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
        for (int j = 0; j < count; ++j) {
            const Eigen::Vector2d error = Y.col(j) - innovation;
            scatter += error * error.transpose();
        }
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H.leftCols<2>() = Eigen::Matrix2d::Identity();
        const double extent_mean_offset = 2.0 * extent_dim + 2.0;
        Eigen::Matrix2d X = predicted.V().template cast<double>() /
            std::max<double>(predicted.v() - extent_mean_offset, 1e-12);
        Eigen::Matrix2d R = rho_ * X;
        if (this->measurement_noise_.rows() == 2) R += this->measurement_noise_;
        Eigen::Matrix2d S = H * predicted.covariance().template cast<double>() * H.transpose()
                          + R / static_cast<double>(count);
        Eigen::Matrix<double, 4, 2> K = predicted.covariance().template cast<double>()
                                      * H.transpose() * S.ldlt().solve(Eigen::Matrix2d::Identity());
        const Eigen::Vector4d delta = K * innovation;
        const Eigen::Vector3d next_r = usggiw::apply_position_error(reference, delta.head<2>());
        const Eigen::Matrix2d T = usggiw::transport_matrix(reference, next_r);
        const Eigen::Vector2d next_w = T * (
            predicted.angular_velocity().template cast<double>() + delta.tail<2>());
        Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d P = (I - K * H) * predicted.covariance().template cast<double>()
                          * (I - K * H).transpose() + K * (R / count) * K.transpose();
        const Eigen::Matrix4d G = usggiw::state_transport_matrix(reference, next_r);
        P = G * P * G.transpose();
        P = 0.5 * (P + P.transpose());

        const Eigen::Matrix2d sqrt_X = sqrtm_spd(X);
        const Eigen::Matrix2d sqrt_R = sqrtm_spd(R);
        const Eigen::Matrix2d sqrt_S = sqrtm_spd(S);
        const Eigen::Matrix2d sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::Matrix2d::Identity());
        const Eigen::Matrix2d sqrt_S_inv = sqrt_S.ldlt().solve(Eigen::Matrix2d::Identity());
        const Eigen::Matrix2d N_hat = sqrt_X * sqrt_S_inv
            * (innovation * innovation.transpose())
            * sqrt_S_inv.transpose() * sqrt_X.transpose();
        const Eigen::Matrix2d Z_hat = sqrt_X * sqrt_R_inv * scatter
            * sqrt_R_inv.transpose() * sqrt_X.transpose();
        const double nv = predicted.v() + count;
        const Eigen::Matrix2d nV_reference =
            predicted.V().template cast<double>() + Z_hat + N_hat;
        const Eigen::Matrix2d nV = T * nV_reference * T.transpose();

        const double v0 = predicted.v();
        const double v1 = nv;
        const double a0 = predicted.alpha();
        const double a1 = a0 + count;
        const double b0 = predicted.beta();
        const double b1 = b0 + 1.0;
        const double log_det_V0 = logdet_spd(predicted.V().template cast<double>());
        const double log_det_V1 = logdet_spd(nV_reference);
        const double log_det_X = logdet_spd(X);
        const double log_det_R = logdet_spd(R);
        const double log_det_S = logdet_spd(S);
        double multivariate_gamma_ratio = 0.0;
        for (int i = 0; i < extent_dim; ++i) {
            multivariate_gamma_ratio += std::lgamma(
                (v1 - extent_dim - 1.0) / 2.0 - static_cast<double>(i) / 2.0);
            multivariate_gamma_ratio -= std::lgamma(
                (v0 - extent_dim - 1.0) / 2.0 - static_cast<double>(i) / 2.0);
        }
        const double log_likelihood =
            (v0 - extent_dim - 1.0) / 2.0 * log_det_V0
          - (v1 - extent_dim - 1.0) / 2.0 * log_det_V1
          + multivariate_gamma_ratio
          + 0.5 * count * log_det_X
          - 0.5 * (count - 1) * log_det_R
          - 0.5 * log_det_S
          + std::lgamma(a1) - std::lgamma(a0)
          + a0 * std::log(b0) - a1 * std::log(b1)
          - (count * std::log(M_PI) + std::log(static_cast<double>(count)))
            * extent_dim / 2.0;
        return {Dist(predicted.alpha() + count, predicted.beta() + 1.0,
                     next_r.template cast<Scalar>(), next_w.template cast<Scalar>(), P.template cast<Scalar>(),
                     nv, nV.template cast<Scalar>()), safe_exp(log_likelihood)};
    }

    [[nodiscard]] double gate(const typename Base::MeasVector& measurement,
                              const Dist& predicted) const override {
        Eigen::Vector3d z = usggiw::normalized(measurement.head(3));
        Eigen::Vector2d e = usggiw::local_error(predicted.direction().template cast<double>(), z);
        Eigen::Matrix2d S = predicted.covariance().template cast<double>().template topLeftCorner<2, 2>();
        const double extent_mean_offset = 2.0 * predicted.extent_dim() + 2.0;
        const Eigen::Matrix2d X = predicted.V().template cast<double>() /
            std::max<double>(predicted.v() - extent_mean_offset, 1e-12);
        S += rho_ * X;
        if (this->measurement_noise_.rows() == 2) S += this->measurement_noise_;
        return e.dot(S.ldlt().solve(e));
    }

    [[nodiscard]] CorrectionResult correct_component(const Dist& z, const Dist& predicted) const {
        const Eigen::Vector3d reference = predicted.direction().template cast<double>();
        const Eigen::Matrix4d Pg = predicted.covariance().template cast<double>();
        const Eigen::Vector3d z_direction = z.direction().template cast<double>();
        const Eigen::Matrix2d Tz = usggiw::transport_matrix(z_direction, reference);
        const Eigen::Matrix4d Gz = usggiw::state_transport_matrix(z_direction, reference);
        const Eigen::Matrix4d Pz = Gz * z.covariance().template cast<double>() * Gz.transpose();
        const Eigen::Matrix2d Vz = Tz * z.V().template cast<double>() * Tz.transpose();
        const double w = std::clamp(ci_weight_, 0.0, 1.0);
        Eigen::Vector4d xg, xz;
        xg << 0.0, 0.0, predicted.angular_velocity().template cast<double>();
        xz.head<2>() = usggiw::local_error(reference, z.direction().template cast<double>());
        xz.tail<2>() = Tz * z.angular_velocity().template cast<double>();
        const Eigen::Matrix4d Jg = Pg.ldlt().solve(Eigen::Matrix4d::Identity());
        const Eigen::Matrix4d Jz = Pz.ldlt().solve(Eigen::Matrix4d::Identity());
        const Eigen::Matrix4d J = w * Jg + (1.0 - w) * Jz;
        const Eigen::Matrix4d P_reference = J.ldlt().solve(Eigen::Matrix4d::Identity());
        const Eigen::Vector4d x = P_reference * (w * Jg * xg + (1.0 - w) * Jz * xz);
        const Eigen::Matrix4d S = Pg + Pz;
        const Eigen::Vector4d innovation = xz - xg;
        const double logdet = S.ldlt().vectorD().array().abs().log().sum();
        const double maha = innovation.dot(S.ldlt().solve(innovation));
        const double likelihood = std::exp(-0.5 * (maha + logdet + 4.0 * std::log(2.0 * M_PI)));
        const Eigen::Vector3d direction = usggiw::apply_position_error(reference, x.head<2>());
        const Eigen::Matrix2d T = usggiw::transport_matrix(reference, direction);
        const Eigen::Matrix4d G = usggiw::state_transport_matrix(reference, direction);
        const Eigen::Matrix4d P = G * P_reference * G.transpose();
        const Eigen::Matrix2d V_reference =
            w * predicted.V().template cast<double>() + (1.0 - w) * Vz;
        return {Dist(w * predicted.alpha() + (1.0 - w) * z.alpha(),
                     w * predicted.beta() + (1.0 - w) * z.beta(),
                     direction.template cast<Scalar>(),
                     (T * x.tail<2>()).template cast<Scalar>(), P.template cast<Scalar>(),
                     w * predicted.v() + (1.0 - w) * z.v(),
                     (T * V_reference * T.transpose()).template cast<Scalar>()), likelihood};
    }

    [[nodiscard]] double gate_component(const Dist& z, const Dist& predicted) const {
        Eigen::Vector4d innovation;
        innovation.head<2>() = usggiw::local_error(
            predicted.direction().template cast<double>(), z.direction().template cast<double>());
        const Eigen::Matrix2d Tz = usggiw::transport_matrix(
            z.direction().template cast<double>(),
            predicted.direction().template cast<double>());
        innovation.tail<2>() = Tz * z.angular_velocity().template cast<double>()
                             - predicted.angular_velocity().template cast<double>();
        const Eigen::Matrix4d Gz = usggiw::state_transport_matrix(
            z.direction().template cast<double>(), predicted.direction().template cast<double>());
        Eigen::Matrix4d S = predicted.covariance().template cast<double>()
                          + Gz * z.covariance().template cast<double>() * Gz.transpose();
        return innovation.dot(S.ldlt().solve(innovation));
    }

    void set_temporal_decay(double value) { eta_ = brew::detail::positive_finite(value, "Temporal decay"); }
    void set_forgetting_factor(double value) { tau_ = brew::detail::positive_finite(value, "Forgetting time"); }
    void set_scaling_parameter(double value) { rho_ = brew::detail::positive_finite(value, "Extent scale"); }
    void set_ci_weight(double value) { ci_weight_ = brew::detail::unit_interval(value, "CI weight"); }

private:
    static Eigen::Matrix2d sqrtm_spd(const Eigen::Matrix2d& matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(
            0.5 * (matrix + matrix.transpose()));
        Eigen::Vector2d eigenvalues = solver.eigenvalues().cwiseMax(
            std::numeric_limits<double>::epsilon()).cwiseSqrt();
        return solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
    }

    static double logdet_spd(const Eigen::Matrix2d& matrix) {
        return matrix.ldlt().vectorD().array().abs()
            .max(std::numeric_limits<double>::min()).log().sum();
    }

    static double safe_exp(double value) {
        return std::exp(std::clamp(value, -700.0, 600.0));
    }

    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;
    double ci_weight_ = 0.5;
};

template<typename Scalar, int De>
struct default_filter<models::USGGIW<Scalar, De>> { using type = USGGIWEKF<Scalar, De>; };

}  // namespace brew::filters
