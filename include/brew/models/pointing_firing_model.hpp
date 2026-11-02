#pragma once

#include "brew/models/template_pose.hpp"
#include "brew/models/categorical_intent.hpp"
#include "brew/models/impulse_prior.hpp"
#include "brew/utils/assert.hpp"
#include <array>
#include <cmath>
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brew::models {


namespace intent_detail {
inline std::optional<Eigen::Vector3d> axis_turn(const Eigen::Vector3d& start,
                                              const Eigen::Vector3d& end) {
    const Eigen::Vector3d cross = start.cross(end);
    const double sine = cross.norm(), cosine = std::clamp(start.dot(end), -1.0, 1.0);
    if (sine < 1e-10) {
        if (cosine <= 0) return std::nullopt;
        return cross;
    }
    return std::acos(cosine) * cross / sine;
}

inline double gaussian_log_density(const Eigen::Vector2d& residual, const Eigen::Matrix2d& covariance) {
    return -0.5 * (residual.dot(covariance.ldlt().solve(residual))
                    + std::log(covariance.determinant()) + 2 * std::log(2 * M_PI));
}

inline double gain_log_evidence(const Eigen::Vector2d& observed, const Eigen::Vector2d& error,
                               const Eigen::Matrix2d& covariance, double maximum) {
    Eigen::LLT<Eigen::Matrix2d> factor(covariance);
    const Eigen::Vector2d y = factor.matrixL().solve(observed);
    const Eigen::Vector2d v = factor.matrixL().solve(maximum * error);
    const double linear = y.dot(v), quadratic = v.squaredNorm();
    const Eigen::Matrix2d lower = factor.matrixL();
    const double normalizer = 2 * std::log(2 * M_PI) + 2 * lower.diagonal().array().log().sum();
    const double zero = -0.5 * (y.squaredNorm() + normalizer);
    if (std::max(std::abs(linear), quadratic) < 0.05) {
        double previous = 1, current = linear, integral = 1 + linear / 2;
        for (int order = 2; order < 18; ++order) {
            const double next = (linear * current - quadratic * previous) / order;
            integral += next / (order + 1);
            previous = current;
            current = next;
        }
        return zero + std::log(integral);
    }
    const double root = std::sqrt(quadratic);
    const double mass = utils::firing_detail::log_normal_mass(-linear / root, root - linear / root);
    const Eigen::Vector2d residual = y - v * (linear / quadratic);
    return -0.5 * (residual.squaredNorm() + normalizer)
        + 0.5 * std::log(2 * M_PI / quadratic) + mass;
}

} // namespace intent_detail

enum class PointingRule { Fixed, Radial, Tangential };

class PointingFiringModel final : public PredictiveBehaviorModel<TemplatePose<>> {
public:
    PointingFiringModel() = default;
    PointingFiringModel(PointingRule rule, const ImpulsePrior& impulse,
                       Eigen::Vector3d axis = Eigen::Vector3d::UnitX(), double radial_sign = 1.0,
                       std::optional<double> firing_probability = std::nullopt,
                       double attitude_sigma = 4.5 * M_PI / 180.0, double gain_max = 0.00575,
                       double holding_probability = 0.2, bool use_direction = true, bool use_impulse = true)
        : pointing(rule), axis(std::move(axis)), radial_sign(radial_sign),
          firing_probability(firing_probability.value_or(impulse.probability)),
          attitude_sigma(attitude_sigma), gain_max(gain_max), holding_probability(holding_probability),
          impulse_max(impulse.maximum), use_direction(use_direction), use_impulse(use_impulse) {
        validate();
    }

    PointingRule pointing = PointingRule::Fixed;
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
    double radial_sign = 1.0;
    double firing_probability = 0.0;

    double attitude_sigma = 4.5 * M_PI / 180.0, gain_max = 0.00575;
    double holding_probability = 0.2, impulse_max = 0.04;
    bool use_direction = true, use_impulse = true;

    void validate() const override {
        for (double value : {attitude_sigma, gain_max, holding_probability, impulse_max})
            if (!std::isfinite(value)) throw std::invalid_argument("Behavior settings must be finite");
        if (attitude_sigma <= 0 || gain_max <= 0 || impulse_max <= 0
            || holding_probability < 0 || holding_probability > 1)
            throw std::invalid_argument("Invalid pointing/firing settings");
        if (!std::isfinite(firing_probability) || firing_probability < 0 || firing_probability > 1)
            throw std::invalid_argument("Intent firing probability must be in [0, 1]");
        if (pointing != PointingRule::Fixed && pointing != PointingRule::Radial
            && pointing != PointingRule::Tangential) throw std::invalid_argument("Unknown pointing rule");
        if (!axis.allFinite() || axis.norm() == 0)
            throw std::invalid_argument("Pointing axis must be finite and nonzero");
        if (radial_sign != -1.0 && radial_sign != 1.0)
            throw std::invalid_argument("Radial sign must be -1 or +1");
    }

    Eigen::Vector3d boresight(const Eigen::Vector3d& position) const {
        if (pointing == PointingRule::Fixed) return axis.normalized();
        const Eigen::Vector3d radial = position / std::max(position.norm(), 1e-9);
        if (pointing == PointingRule::Radial) return radial_sign * radial;
        const Eigen::Vector3d tangent = axis.cross(radial);
        return tangent / std::max(tangent.norm(), 1e-9);
    }
    std::array<double, 2> branch_log_likelihoods(
        const models::TemplatePose<>& pose, const models::TemplatePose<>* previous, double dt) const {
        if (!previous || !use_direction) return {0.0, 0.0};
        const Eigen::Vector3d start = previous->rotation().col(0), end = pose.rotation().col(0);
        const auto turn = intent_detail::axis_turn(start, end);
        if (!turn) return {0.0, 0.0};
        const Eigen::Matrix<double, 3, 2> basis = previous->rotation().rightCols<2>();
        const Eigen::Vector2d observed = basis.transpose() * *turn;
        const Eigen::Matrix3d old_cov = previous->covariance().bottomRightCorner<3, 3>();
        const Eigen::Matrix2d covariance = basis.transpose()
            * (old_cov + pose.covariance().bottomRightCorner<3, 3>()) * basis
            + attitude_sigma * attitude_sigma * Eigen::Matrix2d::Identity();
        const double zero = intent_detail::gaussian_log_density(observed, covariance);
        const double concentration = 1 / (attitude_sigma * attitude_sigma
            + 0.5 * (old_cov.trace() - start.dot(old_cov * start)));
        const double constant = std::log(concentration) - std::log(2 * M_PI)
            - std::log(-std::expm1(-2 * concentration));
        const auto desired = boresight(previous->position());
        const auto error = intent_detail::axis_turn(start, desired);
        const double turning = error ? intent_detail::gain_log_evidence(observed, dt * (basis.transpose() * *error),
                                              covariance, gain_max) - std::log(4 * M_PI) : -INFINITY;
        const double holding = zero + constant + concentration * (std::clamp(start.dot(desired), -1.0, 1.0) - 1);
        return {turning, holding};
    }

    double impulse_log_likelihood(double magnitude, double variance,
                            std::optional<std::array<double, 2>> evidence = std::nullopt) const {
        if (!use_impulse || !std::isfinite(magnitude)) return 0.0;
        BREW_ASSERT(variance > 0, "Impulse likelihood variance must be positive");
        const auto branch = evidence ? *evidence
            : utils::firing_detail::impulse_log_evidence(magnitude, variance, impulse_max);
        return firing_probability == 0.0 ? branch[0]
            : firing_probability == 1.0 ? branch[1]
            : utils::firing_detail::log_add(std::log1p(-firing_probability) + branch[0],
                                    std::log(firing_probability) + branch[1]);
    }

    std::vector<double> log_evidence(const models::TemplatePose<>& current,
                                    const models::TemplatePose<>* previous, double dt) const override {
        const auto [turn, hold] = branch_log_likelihoods(current, previous, dt);
        const double direction = utils::firing_detail::log_add(std::log1p(-holding_probability) + turn,
                                                        std::log(holding_probability) + hold);
        std::optional<std::array<double, 2>> evidence;
        if (std::isfinite(current.impulse_log_zero()) && std::isfinite(current.impulse_log_positive()))
            evidence = std::array<double, 2>{current.impulse_log_zero(), current.impulse_log_positive()};
        return {direction, impulse_log_likelihood(current.impulse_observation(), current.impulse_variance(), evidence)};
    }
};



} // namespace brew::models
