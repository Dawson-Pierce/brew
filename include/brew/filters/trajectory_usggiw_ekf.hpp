#pragma once

#include "brew/filters/filter_base.hpp"
#include "brew/filters/filter_traits.hpp"
#include "brew/utils/parameter_validation.hpp"
#include "brew/models/trajectory_usggiw.hpp"
#include "brew/utils/unit_sphere.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "brew/multiObjectTrackers/spherical_config.hpp"

namespace brew::filters {

// @mex filter
// @mex_name TrajectoryUSGGIWEKF
// @mex_dist TrajectoryUSGGIW
// @mex_setters temporal_decay:scalar, forgetting_factor:scalar, scaling_parameter:scalar, window_size:int, ci_weight:scalar
template <typename Scalar = double, int De = Eigen::Dynamic>
class TrajectoryUSGGIWEKF : public Filter<models::TrajectoryUSGGIW<Scalar, De>> {
public:
    using InnerDist = models::USGGIW<Scalar, De>;
    using Dist = models::TrajectoryUSGGIW<Scalar, De>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;

    void set_application_config(const multiObjectTrackers::SphericalPHDConfig& config, double measurement_sigma,
            double birth_position_variance, double birth_rate_variance, int window,
            double dedup_deg, double velocity_sigma) {
        config_ = config;
        application_setup_ = true;
        window_size_ = window;
        dedup_angle_ = utils::sphere::radians(dedup_deg);
        const double position = utils::sphere::angular_variance(measurement_sigma);
        const double velocity = utils::sphere::angular_variance(velocity_sigma);
        report_covariance_ = Eigen::Vector4d(position, position, velocity, velocity).asDiagonal();
        const double bp = utils::sphere::angular_variance(std::sqrt(birth_position_variance));
        const double bv = utils::sphere::angular_variance(std::sqrt(birth_rate_variance));
        birth_covariance_ = Eigen::Vector4d(bp, bp, bv, bv).asDiagonal();
    }
    const multiObjectTrackers::SphericalPHDConfig& application_config() const { return config_; }
    bool groups_observations() const override { return false; }
    std::unique_ptr<models::Mixture<Dist>> prepare_step(Eigen::MatrixXd& measurements,
            const models::Mixture<Dist>& previous, double dt, double prune_threshold,
            double gate_threshold, std::unique_ptr<models::Mixture<InnerDist>>& input) const override {
        if (!application_setup_ || !input) return nullptr;
        auto reports = std::make_unique<models::Mixture<InnerDist>>();
        auto births = std::make_unique<models::Mixture<Dist>>();
        const auto predicted = predicted_components(previous, dt, prune_threshold);
        measurements.resize(3, input->size());
        for (std::size_t i = 0; i < input->size(); ++i) {
            const auto report = augmented(input->component(i), report_covariance_);
            reports->add_component(report, 1.);
            measurements.col(i) = report.direction().template cast<double>();
            if (!explained(report, predicted, gate_threshold))
                births->add_component(Dist(4, augmented(report, birth_covariance_), window_size_), config_.birth_weight);
        }
        input = std::move(reports);
        return births;
    }
    void select_estimates(std::vector<std::size_t>& indices,
            const std::vector<const Dist*>& states, const std::vector<double>& weights) const override {
        if (dedup_angle_ <= 0. || indices.size() < 2) return;
        std::stable_sort(indices.begin(), indices.end(), [&](auto a, auto b) { return weights[a] > weights[b]; });
        std::vector<std::size_t> kept;
        const double limit = std::cos(dedup_angle_);
        for (auto i : indices) {
            const auto& direction = states[i]->current().direction();
            if (std::all_of(kept.begin(), kept.end(), [&](auto j) {
                    return direction.dot(states[j]->current().direction()) <= limit;
                })) kept.push_back(i);
        }
        indices = std::move(kept);
    }
    Eigen::MatrixXd history(const Dist& trajectory) const override {
        const Eigen::MatrixXd stored = trajectory.state_history_directions().template cast<double>().transpose();
        const Eigen::MatrixXd window = trajectory.window_directions().template cast<double>().transpose();
        if (!window.rows()) return stored;
        Eigen::Index prefix = std::max<Eigen::Index>(0, stored.rows()-window.rows()+1);
        if (window.rows() == 1 && stored.rows() == 1) prefix = 0;
        Eigen::MatrixXd path(prefix+window.rows(), 3);
        path.topRows(prefix) = stored.topRows(prefix);
        path.bottomRows(window.rows()) = window;
        return path;
    }
    double detection_probability(const Dist& trajectory, const Eigen::MatrixXd& field,
            double radius, double pd, double pd_out, const std::vector<int>& groups = {}) const override {
        if (radius <= 0.0 || field.cols() == 0)
            return pd;
        const Eigen::Vector3d direction =
            trajectory.current().direction().template cast<double>();
        std::vector<int> visible_groups;
        const double cosine_radius = std::cos(radius);
        for (int i = 0; i < field.cols(); ++i) {
            const Eigen::Vector3d center = utils::sphere::normalized(field.col(i));
            if (direction.dot(center) < cosine_radius) continue;
            const int group = groups.empty()
                ? i : groups[static_cast<std::size_t>(i)];
            if (std::find(visible_groups.begin(), visible_groups.end(), group)
                    == visible_groups.end())
                visible_groups.push_back(group);
        }
        double missed = 1.0;
        for (std::size_t i = 0; i < visible_groups.size(); ++i)
            missed *= 1.0 - pd;
        return std::max(pd_out, 1.0 - missed);
    }

    std::vector<typename Base::WeightedDensity> missed_detection(
            const Dist& prior, double pd, double weight) const override {
        std::vector<typename Base::WeightedDensity> result;
        if (weight * (1. - pd) > 0.) result.push_back({prior, weight * (1. - pd)});
        const auto& rate = models::current_density(prior);
        const double q0 = std::pow(rate.beta() / (rate.beta() + 1.), rate.alpha());
        if (weight * pd * q0 > 0.) {
            auto updated = prior;
            models::current_density(updated).beta() += 1.;
            result.push_back({std::move(updated), weight * pd * q0});
        }
        return result;
    }

    typename Base::CorrectionResult correct_report(
            const typename Base::Report& report, const Dist& prior) const override {
        return correct_component(report, prior);
    }
    double gate_report(const typename Base::Report& report, const Dist& prior) const override {
        return gate_component(report, prior);
    }

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto out = std::make_unique<TrajectoryUSGGIWEKF>();
        out->process_noise_ = this->process_noise_;
        out->measurement_noise_ = this->measurement_noise_;
        out->eta_ = eta_;
        out->tau_ = tau_;
        out->rho_ = rho_;
        out->window_size_ = window_size_;
        out->ci_weight_ = ci_weight_;
        out->config_ = config_;
        out->application_setup_ = application_setup_;
        out->dedup_angle_ = dedup_angle_;
        out->report_covariance_ = report_covariance_;
        out->birth_covariance_ = birth_covariance_;

        return out;
    }

    [[nodiscard]] Dist predict(double dt, const Dist& previous) const override {
        Dist result = previous;
        const auto& prior = previous.current();
        const Eigen::Vector3d direction = prior.direction().template cast<double>();
        const Eigen::Vector2d velocity = prior.angular_velocity().template cast<double>();
        const utils::sphere::PropagatedState next = utils::sphere::propagate(direction, velocity, dt);

        const Eigen::Matrix4d F = utils::sphere::transition_jacobian(
            direction, velocity, dt, next);
        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        if (this->process_noise_.rows() == 4) Q = this->process_noise_.template cast<double>();

        const int extent_dim = prior.extent_dim();
        const double floor = 2.0 * extent_dim + 2.0;
        const double next_v = floor + std::exp(-dt / tau_) * (prior.v() - floor);
        const double extent_scale = (next_v - floor) / std::max<double>(prior.v() - floor, 1e-12);

        result.advance_window();
        const int last = result.last_index();
        const int prior_last = last - 1;
        if (prior_last >= 0) {
            for (int i = 0; i < last; ++i) {
                result.cov_at(last, i) = F * result.cov_at(prior_last, i);
                result.cov_at(i, last) = result.cov_at(last, i).transpose();
            }
            result.cov_at(last, last) =
                F * result.cov_at(prior_last, prior_last) * F.transpose() + Q;
        } else {
            result.cov_at(last, last) = Q;
        }

        result.mean_at(last).setZero();
        result.mean_at(last).template tail<2>() = next.velocity.template cast<Scalar>();
        const Eigen::Matrix2d extent_transport =
            utils::sphere::propagation_extent_transport(direction, next);
        result.history_at(last) = InnerDist(
            prior.alpha() / eta_, prior.beta() / eta_, next.direction.template cast<Scalar>(),
            next.velocity.template cast<Scalar>(), result.cov_at(last, last), next_v,
            (extent_scale * extent_transport * prior.V().template cast<double>()
                * extent_transport.transpose()).template cast<Scalar>());
        return result;
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement, const Dist& predicted) const override {
        if (measurement.size() % 3 != 0 || measurement.size() == 0)
            return {predicted, 0.0};
        const int count = static_cast<int>(measurement.size()) / 3;
        Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> Z(
            measurement.data(), 3, count);
        const auto& current = predicted.current();
        const Eigen::Vector3d reference = current.direction().template cast<double>();
        Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, count);
        for (int i = 0; i < count; ++i)
            Y.col(i) = utils::sphere::local_error(reference, Z.col(i));
        const Eigen::Vector2d innovation = Y.rowwise().mean();
        Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
        for (int i = 0; i < count; ++i) {
            const Eigen::Vector2d error = Y.col(i) - innovation;
            scatter += error * error.transpose();
        }

        const int live = predicted.stacked_size();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, live);
        H.block<2, 2>(0, live - 4) = Eigen::Matrix2d::Identity();
        const double extent_mean_offset = 2.0 * current.extent_dim() + 2.0;
        const Eigen::Matrix2d extent = current.V().template cast<double>() /
            std::max<double>(current.v() - extent_mean_offset, 1e-12);
        Eigen::Matrix2d R = rho_ * extent;
        if (this->measurement_noise_.rows() == 2) R += this->measurement_noise_;
        const Eigen::MatrixXd P = predicted.covariance().topLeftCorner(live, live);
        const Eigen::Matrix2d S = H * P * H.transpose() + R / static_cast<double>(count);
        const Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::Matrix2d::Identity());
        const Eigen::VectorXd delta = K * innovation;
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(live, live);

        const Eigen::Matrix2d sqrt_extent = sqrtm_spd(extent);
        const Eigen::Matrix2d sqrt_R = sqrtm_spd(R);
        const Eigen::Matrix2d sqrt_S = sqrtm_spd(S);
        const Eigen::Matrix2d sqrt_R_inv = sqrt_R.ldlt().solve(Eigen::Matrix2d::Identity());
        const Eigen::Matrix2d sqrt_S_inv = sqrt_S.ldlt().solve(Eigen::Matrix2d::Identity());
        const Eigen::Matrix2d N_hat = sqrt_extent * sqrt_S_inv
            * (innovation * innovation.transpose())
            * sqrt_S_inv.transpose() * sqrt_extent.transpose();
        const Eigen::Matrix2d Z_hat = sqrt_extent * sqrt_R_inv * scatter
            * sqrt_R_inv.transpose() * sqrt_extent.transpose();
        const Eigen::Matrix2d next_V_reference =
            current.V().template cast<double>() + Z_hat + N_hat;

        Dist result = predicted;
        result.covariance().topLeftCorner(live, live) =
            (I - K * H) * P * (I - K * H).transpose()
            + K * (R / static_cast<double>(count)) * K.transpose();
        apply_delta(result, delta);

        auto& updated = result.current();
        updated.alpha() = current.alpha() + count;
        updated.beta() = current.beta() + 1.0;
        updated.v() = current.v() + count;
        const Eigen::Matrix2d reset = utils::sphere::transport_matrix(
            reference, updated.direction().template cast<double>());
        updated.V() = (reset * next_V_reference * reset.transpose()).template cast<Scalar>();

        constexpr int extent_dimension = 2;
        const double v0 = current.v();
        const double v1 = updated.v();
        double multivariate_gamma_ratio = 0.0;
        for (int i = 0; i < extent_dimension; ++i) {
            multivariate_gamma_ratio += std::lgamma(
                (v1 - extent_dimension - 1.0) / 2.0 - static_cast<double>(i) / 2.0);
            multivariate_gamma_ratio -= std::lgamma(
                (v0 - extent_dimension - 1.0) / 2.0 - static_cast<double>(i) / 2.0);
        }
        const double log_likelihood =
            (v0 - extent_dimension - 1.0) / 2.0
                * logdet_spd(current.V().template cast<double>())
          - (v1 - extent_dimension - 1.0) / 2.0 * logdet_spd(next_V_reference)
          + multivariate_gamma_ratio
          + 0.5 * count * logdet_spd(extent)
          - 0.5 * (count - 1) * logdet_spd(R)
          - 0.5 * logdet_spd(S)
          + std::lgamma(updated.alpha()) - std::lgamma(current.alpha())
          + current.alpha() * std::log(current.beta())
          - updated.alpha() * std::log(updated.beta())
          - (count * std::log(M_PI) + std::log(static_cast<double>(count)))
            * extent_dimension / 2.0;
        return {std::move(result), safe_exp(log_likelihood)};
    }

    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement, const Dist& predicted) const override {
        if (measurement.size() < 3) return std::numeric_limits<double>::infinity();
        const Eigen::Vector3d z = utils::sphere::normalized(measurement.head(3));
        const Eigen::Vector2d error = utils::sphere::local_error(
            predicted.current().direction().template cast<double>(), z);
        Eigen::Matrix2d S = predicted.get_last_cov().template topLeftCorner<2, 2>();
        const auto& current = predicted.current();
        const double extent_mean_offset = 2.0 * current.extent_dim() + 2.0;
        S += rho_ * current.V().template cast<double>()
           / std::max<double>(current.v() - extent_mean_offset, 1e-12);
        if (this->measurement_noise_.rows() == 2) S += this->measurement_noise_;
        return error.dot(S.ldlt().solve(error));
    }

    [[nodiscard]] CorrectionResult correct_component(
        const InnerDist& report, const Dist& predicted) const {
        const auto& current = predicted.current();
        const Eigen::Vector3d reference = current.direction().template cast<double>();
        const Eigen::Vector3d report_direction = report.direction().template cast<double>();
        const Eigen::Matrix2d report_transport =
            utils::sphere::transport_matrix(report_direction, reference);
        const Eigen::Matrix4d report_state_transport =
            utils::sphere::state_transport_matrix(report_direction, reference);
        const Eigen::Matrix2d report_extent = report_transport
            * report.V().template cast<double>() * report_transport.transpose();
        Eigen::Vector4d ground = Eigen::Vector4d::Zero();
        Eigen::Vector4d measurement = Eigen::Vector4d::Zero();
        ground.tail<2>() = current.angular_velocity().template cast<double>();
        measurement.head<2>() = utils::sphere::local_error(
            reference, report.direction().template cast<double>());
        measurement.tail<2>() = report_transport
            * report.angular_velocity().template cast<double>();

        Eigen::Matrix4d Pg = predicted.get_last_cov().template cast<double>();
        Eigen::Matrix4d Pz = report_state_transport
            * report.covariance().template cast<double>() * report_state_transport.transpose();
        Pg = 0.5 * (Pg + Pg.transpose());
        Pz = 0.5 * (Pz + Pz.transpose());
        Eigen::LDLT<Eigen::Matrix4d> pg_ldlt(Pg), pz_ldlt(Pz);
        if (pg_ldlt.info() != Eigen::Success || pz_ldlt.info() != Eigen::Success)
            return {predicted, 0.0};
        const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
        const Eigen::Matrix4d Jg = pg_ldlt.solve(identity);
        const Eigen::Matrix4d Jz = pz_ldlt.solve(identity);
        const double omega = std::clamp(ci_weight_, 0.0, 1.0);
        const Eigen::Matrix4d J = omega * Jg + (1.0 - omega) * Jz;
        Eigen::LDLT<Eigen::Matrix4d> j_ldlt(J);
        if (j_ldlt.info() != Eigen::Success) return {predicted, 0.0};
        const Eigen::Matrix4d Pf = j_ldlt.solve(identity);
        const Eigen::Vector4d fused = Pf * (
            omega * Jg * ground + (1.0 - omega) * Jz * measurement);

        const int live = predicted.stacked_size();
        const int current_offset = live - 4;
        const Eigen::MatrixXd P = predicted.covariance().topLeftCorner(live, live);
        Eigen::MatrixXd next_P = P;
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(live);
        const Eigen::Vector4d current_delta = fused - ground;
        if (current_offset > 0) {
            const Eigen::MatrixXd cross = P.block(0, current_offset, current_offset, 4);
            const Eigen::MatrixXd A = cross * Jg;
            delta.head(current_offset) = A * current_delta;
            next_P.topLeftCorner(current_offset, current_offset) +=
                A * (Pf - Pg) * A.transpose();
            next_P.block(0, current_offset, current_offset, 4) = A * Pf;
            next_P.block(current_offset, 0, 4, current_offset) = Pf * A.transpose();
        }
        delta.tail<4>() = current_delta;
        next_P.bottomRightCorner<4, 4>() = Pf;

        Dist result = predicted;
        result.covariance().topLeftCorner(live, live) = 0.5 * (next_P + next_P.transpose());
        apply_delta(result, delta);
        auto& updated = result.current();
        updated.alpha() = omega * current.alpha() + (1.0 - omega) * report.alpha();
        updated.beta() = omega * current.beta() + (1.0 - omega) * report.beta();
        updated.v() = omega * current.v() + (1.0 - omega) * report.v();
        const Eigen::Matrix2d output_transport = utils::sphere::transport_matrix(
            reference, updated.direction().template cast<double>());
        updated.V() = (output_transport * (
            omega * current.V().template cast<double>() + (1.0 - omega) * report_extent)
            * output_transport.transpose()).template cast<Scalar>();
        const Eigen::Matrix4d S = Pg + Pz;
        Eigen::LDLT<Eigen::Matrix4d> s_ldlt(S);
        if (s_ldlt.info() != Eigen::Success) return {predicted, 0.0};
        const Eigen::Vector4d residual = measurement - ground;
        const double logdet = s_ldlt.vectorD().array().abs().log().sum();
        const double mahalanobis = residual.dot(s_ldlt.solve(residual));
        const double likelihood = std::exp(
            -0.5 * (mahalanobis + logdet + 4.0 * std::log(2.0 * M_PI)));
        return {std::move(result), likelihood};
    }

    [[nodiscard]] double gate_component(const InnerDist& report, const Dist& predicted) const {
        const Eigen::Vector2d position_error = utils::sphere::local_error(
            predicted.current().direction().template cast<double>(),
            report.direction().template cast<double>());
        const Eigen::Matrix4d report_transport = utils::sphere::state_transport_matrix(
            report.direction().template cast<double>(),
            predicted.current().direction().template cast<double>());
        const Eigen::Matrix4d report_covariance = report_transport
            * report.covariance().template cast<double>() * report_transport.transpose();
        Eigen::Vector4d residual;
        residual.head<2>() = position_error;
        const Eigen::Matrix2d velocity_transport = utils::sphere::transport_matrix(
            report.direction().template cast<double>(),
            predicted.current().direction().template cast<double>());
        residual.tail<2>() = velocity_transport
                               * report.angular_velocity().template cast<double>()
                           - predicted.current().angular_velocity().template cast<double>();
        const Eigen::Matrix4d S = predicted.get_last_cov().template cast<double>()
                              + report_covariance;
        return residual.dot(S.ldlt().solve(residual));
    }

    void set_temporal_decay(double value) { eta_ = brew::utils::detail::positive_finite(value, "Temporal decay"); }
    void set_forgetting_factor(double value) { tau_ = brew::utils::detail::positive_finite(value, "Forgetting time"); }
    void set_scaling_parameter(double value) { rho_ = brew::utils::detail::positive_finite(value, "Extent scale"); }
    void set_window_size(int value) { window_size_ = value > 0 ? value : 1; }
    void set_ci_weight(double value) { ci_weight_ = brew::utils::detail::unit_interval(value, "CI weight"); }

private:
    static InnerDist augmented(const InnerDist& state, const Eigen::Matrix4d& addition) {
        return InnerDist(state.alpha(), state.beta(), state.direction(), state.angular_velocity(),
                     state.covariance()+addition.template cast<Scalar>(), state.v(), state.V());
    }
    struct Predicted {
        Eigen::Vector3d direction;
        Eigen::Matrix4d covariance;
    };

    std::vector<Predicted> predicted_components(const models::Mixture<Dist>& latest, double dt, double prune_threshold) const {
        std::vector<Predicted> result;
        Eigen::Matrix4d process_covariance = Eigen::Matrix4d::Zero();
        if (this->process_noise_.rows() == 4)
            process_covariance = this->process_noise_.template cast<double>();
        for (std::size_t i = 0; i < latest.size(); ++i) {
            if (latest.weight(i) < prune_threshold) continue;
            const auto& state = latest.component(i).current();
            const auto next = utils::sphere::propagate(state.direction().template cast<double>(), state.angular_velocity().template cast<double>(), dt);
            const Eigen::Matrix4d F = utils::sphere::transition_jacobian(state.direction().template cast<double>(), state.angular_velocity().template cast<double>(), dt);
            result.push_back({next.direction, F*state.covariance().template cast<double>()*F.transpose()+process_covariance});
        }
        return result;
    }

    bool explained(const InnerDist& report, const std::vector<Predicted>& predicted, double gate_threshold) const {
        for (const auto& state : predicted) {
            const Eigen::Vector2d residual = utils::sphere::local_error(state.direction, report.direction().template cast<double>());
            Eigen::Matrix2d innovation = state.covariance.template topLeftCorner<2,2>()
                                      + report.covariance().template topLeftCorner<2,2>().template cast<double>();
            innovation = (.5*(innovation+innovation.transpose())).eval();
            innovation.diagonal().array() += 1e-12;
            if (residual.dot(innovation.partialPivLu().solve(residual)) < gate_threshold) return true;
        }
        return false;
    }

    multiObjectTrackers::SphericalPHDConfig config_;
    bool application_setup_ = false;
    double dedup_angle_ = 0.;
    Eigen::Matrix4d report_covariance_ = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d birth_covariance_ = Eigen::Matrix4d::Zero();

    static Eigen::Matrix2d sqrtm_spd(const Eigen::Matrix2d& matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(
            0.5 * (matrix + matrix.transpose()));
        const Eigen::Vector2d values = solver.eigenvalues().cwiseMax(
            std::numeric_limits<double>::epsilon()).cwiseSqrt();
        return solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
    }

    static double logdet_spd(const Eigen::Matrix2d& matrix) {
        return matrix.ldlt().vectorD().array().abs()
            .max(std::numeric_limits<double>::min()).log().sum();
    }

    static double safe_exp(double value) {
        return std::exp(std::clamp(value, -700.0, 600.0));
    }

    static void apply_delta(Dist& trajectory, const Eigen::VectorXd& delta) {
        const int count = trajectory.window_size();
        const int live = trajectory.stacked_size();
        Eigen::MatrixXd reset = Eigen::MatrixXd::Zero(live, live);
        for (int i = 0; i < count; ++i) {
            auto& state = trajectory.history_at(i);
            const Eigen::Vector4d local = delta.segment<4>(4 * i);
            const Eigen::Vector3d old_direction = state.direction().template cast<double>();
            const Eigen::Vector3d new_direction = utils::sphere::apply_position_error(
                old_direction, local.head<2>());
            const Eigen::Matrix2d T = utils::sphere::transport_matrix(old_direction, new_direction);
            const Eigen::Matrix4d G = utils::sphere::state_transport_matrix(
                old_direction, new_direction);
            reset.block<4, 4>(4 * i, 4 * i) = G;
            state.direction() = new_direction.template cast<Scalar>();
            state.angular_velocity() = (T * (
                state.angular_velocity().template cast<double>() + local.tail<2>()))
                    .template cast<Scalar>();
            state.V() = (T * state.V().template cast<double>() * T.transpose())
                .template cast<Scalar>();
            trajectory.mean_at(i).setZero();
            trajectory.mean_at(i).template tail<2>() = state.angular_velocity();
        }
        Eigen::MatrixXd covariance = trajectory.covariance().topLeftCorner(live, live);
        covariance = reset * covariance * reset.transpose();
        trajectory.covariance().topLeftCorner(live, live) =
            0.5 * (covariance + covariance.transpose());
        for (int i = 0; i < count; ++i)
            trajectory.history_at(i).covariance() = trajectory.cov_at(i, i);
    }

    double eta_ = 1.0;
    double tau_ = 1.0;
    double rho_ = 1.0;
    int window_size_ = Dist::kDefaultWindow;
    double ci_weight_ = 0.5;
};

template <typename Scalar, int De>
struct default_filter<models::TrajectoryUSGGIW<Scalar, De>> {
    using type = TrajectoryUSGGIWEKF<Scalar, De>;
};

}  // namespace brew::filters
