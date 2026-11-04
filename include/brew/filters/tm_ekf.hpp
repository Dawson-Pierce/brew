#pragma once

#include "brew/filters/filter_traits.hpp"

#include "brew/utils/assert.hpp"
#include "brew/filters/filter_base.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/utils/registration/icp_base.hpp"
#include "brew/utils/registration/point_cloud.hpp"
#include "brew/utils/registration/so3.hpp"
#include "brew/utils/registration/template_library.hpp"
#include "brew/utils/registration/tm_icp_runner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <tuple>
#include <optional>

namespace brew::filters {

// @mex filter
// @mex_name TmEkf
// @mex_dist TemplatePose
// @mex_setters rotation_process_noise:mat, surface_sigma_sq:double, surface_outlier_probability:double, surface_outlier_volume:double, pose_point_sigma_sq:double, pose_angular_pitch:double, pose_return_cap:int
// @mex_handle_setters icp:IcpBase, template_library:TemplateLibrary
// @mex_config dynamics, template_library, icp, process_noise, measurement_noise, measurement_jacobian, rotation_process_noise, surface_sigma_sq, surface_outlier_probability, surface_outlier_volume, pose_point_sigma_sq, pose_angular_pitch, pose_return_cap

template <typename Scalar = double, int D = Eigen::Dynamic>
class TmEkf : public Filter<models::TemplatePose<Scalar, D>> {
public:
    using Dist = models::TemplatePose<Scalar, D>;
    using Base = Filter<Dist>;
    using CorrectionResult = typename Base::CorrectionResult;
    using LogCorrectionResult = typename Base::LogCorrectionResult;
    using typename Base::DynamicsType;

    // Unspecified values retain the ordinary constructor's defaults.
    struct Config {
        std::shared_ptr<DynamicsType> dynamics;
        std::shared_ptr<utils::registration::TemplateLibrary> template_library;
        std::shared_ptr<utils::registration::IcpBase> icp;
        std::optional<typename Base::StateMatrix> process_noise;
        std::optional<typename Base::MeasNoiseMatrix> measurement_noise;
        std::optional<typename Base::MeasurementJacobian> measurement_jacobian;
        std::optional<Eigen::MatrixXd> rotation_process_noise;
        std::optional<double> surface_sigma_sq, surface_outlier_probability, surface_outlier_volume;
        std::optional<double> pose_point_sigma_sq, pose_angular_pitch;
        std::optional<int> pose_return_cap;
    };

    TmEkf() = default;
    explicit TmEkf(const Config& c) : TmEkf() {
        if (c.dynamics) this->set_dynamics(c.dynamics);
        if (c.template_library) set_template_library(c.template_library);
        if (c.icp) set_icp(c.icp);
        if (c.process_noise) this->set_process_noise(*c.process_noise);
        if (c.measurement_noise) this->set_measurement_noise(*c.measurement_noise);
        if (c.measurement_jacobian) this->set_measurement_jacobian(*c.measurement_jacobian);
        if (c.rotation_process_noise) set_rotation_process_noise(*c.rotation_process_noise);
        if (c.surface_sigma_sq) set_surface_sigma_sq(*c.surface_sigma_sq);
        if (c.surface_outlier_probability) set_surface_outlier_probability(*c.surface_outlier_probability);
        if (c.surface_outlier_volume) set_surface_outlier_volume(*c.surface_outlier_volume);
        if (c.pose_point_sigma_sq) set_pose_point_sigma_sq(*c.pose_point_sigma_sq);
        if (c.pose_angular_pitch) set_pose_angular_pitch(*c.pose_angular_pitch);
        if (c.pose_return_cap) set_pose_return_cap(*c.pose_return_cap);
    }

    std::vector<typename Base::WeightedDensity> missed_detection(
            const Dist& prior, double pd, double weight) const override {
        auto result = Base::missed_detection(prior, pd, weight);
        for (auto& branch : result)
            if (branch.distribution.firing().firing_components.empty())
                update_firing_report(branch.distribution);
        return result;
    }

    bool log_domain_correction() const override { return true; }
    void expand_gates(std::vector<bool>& accepted, const std::vector<const Dist*>& priors) const override {
        for (std::size_t i = 0; i < priors.size(); ++i) {
            if (!accepted[i] || priors[i]->impulse_parent() < 0) continue;
            for (std::size_t j = 0; j < priors.size(); ++j)
                if (priors[j]->impulse_parent() == priors[i]->impulse_parent()) accepted[j] = true;
        }
    }
    void adjust_log_weights(std::vector<std::unique_ptr<Dist>>& states,
            const std::vector<double>&, std::vector<double>&) const override {
        // Correction already includes each parent's predictive evidence. Share
        // sibling evidence for the marginal firing report without renormalizing it.
        for (auto& state : states) {
            const int parent = state->impulse_parent();
            if (parent < 0) continue;
            double log_zero = -std::numeric_limits<double>::infinity();
            double log_positive = -std::numeric_limits<double>::infinity();
            double observation = std::numeric_limits<double>::quiet_NaN();
            double variance = std::numeric_limits<double>::quiet_NaN();
            for (const auto& sibling : states) {
                if (sibling->impulse_parent() != parent) continue;
                if (sibling->impulse_branch() == 0)
                    log_zero = sibling->impulse_log_likelihood();
                if (sibling->impulse_branch() == 1) {
                    log_positive = sibling->impulse_log_likelihood();
                    observation = sibling->impulse_observation();
                    variance = sibling->impulse_variance();
                }
            }
            state->set_impulse_branch_evidence(log_zero, log_positive);
            state->set_impulse_evidence(observation, variance);
        }
        for (auto& state : states) update_firing_report(*state);
    }

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<TmEkf<Scalar, D>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->rotation_process_noise_ = rotation_process_noise_;
        c->surface_sigma_sq_ = surface_sigma_sq_;
        c->surface_outlier_probability_ = surface_outlier_probability_;
        c->surface_outlier_volume_ = surface_outlier_volume_;
        c->pose_point_sigma_sq_ = pose_point_sigma_sq_;
        c->pose_angular_pitch_ = pose_angular_pitch_;
        c->pose_return_cap_ = pose_return_cap_;
        c->icp_runner_ = icp_runner_;
        return c;
    }

    [[nodiscard]] Dist predict(
        double dt,
        const Dist& prev) const override {
        BREW_ASSERT(impulse_max() == 0.0 || dt <= 0.0,
                    "TmEkf: impulse prediction produces a mixture; use predict_batch_dynamic");
        return predict_branch(dt, prev, 0);
    }

    void predict_batch_dynamic(double dt, models::Mixture<Dist, Eigen::Dynamic>& mix) const override {
        if (this->predict_impulses(dt, mix)) return;
        for (std::size_t i = 0; i < mix.size(); ++i)
            mix.component(i) = predict_branch(dt, mix.component(i), 0);
    }

    Dist predict_impulse_branch(double dt, const Dist& previous,
            const dynamics::ImpulseModel&, int branch, int parent) const override {
        auto predicted = predict_branch(dt, previous, branch);
        predicted.set_impulse_branch(branch, parent);
        update_firing_report(predicted);
        return predicted;
    }

    [[nodiscard]] Dist predict_branch(double dt, const Dist& prev, int branch) const {

        const int n_trans = prev.trans_dim();
        const int n_rot = prev.rot_dim();
        const int n_aug = prev.aug_dim();

        Eigen::MatrixXd F;
        Eigen::VectorXd next_mean;
        Eigen::Matrix3d next_rotation;

        if (n_trans == 9) {
            const Eigen::VectorXd orbital = prev.mean().head(6);
            if (!prev.needs_pca_alignment()) {
                Eigen::MatrixXd rot_dyn = prev.rotation();
                F = this->dyn_obj_->get_state_mat(dt, orbital, rot_dyn);
                next_mean = prev.mean();
                next_mean.head(6) = this->dyn_obj_->propagate_state(dt, orbital, rot_dyn);
            } else {
                F = this->dyn_obj_->get_state_mat(dt, orbital);
                next_mean = prev.mean();
                next_mean.head(6) = this->dyn_obj_->propagate_state(dt, orbital);
            }
            next_rotation = prev.needs_pca_alignment() ? prev.rotation()
                : utils::registration::so3::exp(dt * Eigen::Vector3d(prev.mean().template segment<3>(6)))
                    * prev.rotation();
        } else if (!prev.needs_pca_alignment()) {

            Eigen::MatrixXd rot_dyn = prev.rotation();
            F = this->dyn_obj_->get_state_mat(dt, prev.mean(), rot_dyn);
            next_mean = this->dyn_obj_->propagate_state(dt, prev.mean(), rot_dyn);
            next_rotation = this->dyn_obj_->propagate_extent(dt, prev.mean(), rot_dyn);
        } else {
            F = this->dyn_obj_->get_state_mat(dt, prev.mean());
            next_mean = this->dyn_obj_->propagate_state(dt, prev.mean());
            next_rotation = prev.rotation();
        }

        Eigen::MatrixXd F_aug = Eigen::MatrixXd::Identity(n_aug, n_aug);
        if (n_trans == 9) {
            F_aug.topLeftCorner(6, 6) = F;
            if (!prev.needs_pca_alignment()) {
                const Eigen::Vector3d phi = dt * prev.mean().template segment<3>(6);
                F_aug.block<3, 3>(n_trans, 6) = dt * utils::registration::so3::left_jacobian(phi);
                F_aug.bottomRightCorner<3, 3>() = utils::registration::so3::exp(phi);
            }
        } else {
            F_aug.topLeftCorner(n_trans, n_trans) = F;
        }

        Eigen::MatrixXd Q = this->process_noise_;
        Eigen::VectorXd impulse_response = Eigen::VectorXd::Zero(n_aug);
        if (branch == 1 && dt > 0.0 && impulse_model()) {
            const auto effect = impulse_model()->prediction(
                F_aug.topLeftCorner(n_trans, n_trans), prev.rotation(),
                prev.covariance().template bottomRightCorner<3, 3>());
            impulse_response.head(n_trans) = effect.response;
            next_mean += effect.mean_shift;
            F_aug.block(0, n_trans, n_trans, 3) += effect.attitude_jacobian;
            Q += effect.covariance;
        }
        auto out = predict_with(dt, prev, std::move(F_aug), std::move(next_mean),
                                std::move(next_rotation), Q);
        out.set_impulse_branch(branch);
        out.set_impulse_response(std::move(impulse_response));
        out.set_impulse_evidence(std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN());
        out.set_impulse_log_likelihood(std::numeric_limits<double>::quiet_NaN());
        out.set_impulse_branch_evidence(std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN());
        update_firing_report(out);
        return out;
    }

    [[nodiscard]] CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {
        return correct_with_log(measurement, predicted).first;
    }

    [[nodiscard]] LogCorrectionResult correct_with_log(
            const typename Base::MeasVector& measurement, const Dist& predicted) const override {
        const auto cloud = measurement_cloud(measurement);
        const auto registered = register_cloud(cloud, predicted);
        last_icp_iterations_ = registered.iterations;
        return correct_registered(predicted, registered, surface_score(cloud, predicted.template_id(), registered));
    }

    std::vector<LogCorrectionResult> correct_batch_with_log(
            const typename Base::MeasVector& measurement,
            const std::vector<const Dist*>& priors) const override {
        const auto cloud = measurement_cloud(measurement);
        using Registration = utils::registration::IcpPseudoMeasurement;
        std::unordered_map<int, std::vector<std::pair<Registration, double>>> candidates;
        // Every gated hypothesis contributes a registration start. Birth and
        // continuation priors then compete using the same set of fitted poses.
        for (const auto* prior : priors) {
            auto fit = register_cloud(cloud, *prior);
            auto& fits = candidates[prior->template_id()];
            const bool duplicate = std::any_of(fits.begin(), fits.end(), [&](const auto& item) {
                return (item.first.t-fit.t).norm() < 1e-9 && (item.first.R-fit.R).norm() < 1e-9;
            });
            if (!duplicate) fits.emplace_back(fit, surface_score(cloud, prior->template_id(), fit));
        }
        std::vector<std::vector<LogCorrectionResult>> evaluated;
        for (const auto* prior : priors) {
            std::vector<LogCorrectionResult> alternatives;
            for (const auto& [fit, score] : candidates.at(prior->template_id())) {
                auto registered = fit;
                registered.R_ref = prior->needs_pca_alignment() ? fit.R : prior->rotation();
                alternatives.push_back(correct_registered(*prior, registered, score));
            }
            evaluated.push_back(std::move(alternatives));
        }
        std::vector<LogCorrectionResult> result;
        for (std::size_t i = 0; i < priors.size(); ++i) {
            std::size_t best = 0;
            double best_score = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < evaluated[i].size(); ++k) {
                double score = evaluated[i][k].second;
                if (priors[i]->impulse_parent() >= 0) {
                    score = -std::numeric_limits<double>::infinity();
                    for (std::size_t j = 0; j < priors.size(); ++j) {
                        if (priors[j]->impulse_parent() != priors[i]->impulse_parent()) continue;
                        const double probability = priors[j]->impulse_branch() == 1
                            ? impulse_probability() : 1.0-impulse_probability();
                        score = utils::firing_detail::log_add(score,
                            std::log(probability) + evaluated[j][k].second);
                    }
                }
                if (score > best_score) { best_score = score; best = k; }
            }
            // Keep siblings on one registered pose, selected by their marginal
            // predictive score. This is a dominant-mode approximation, not a sum
            // over duplicate ICP convergence points.
            last_icp_iterations_ = candidates.at(priors[i]->template_id())[best].first.iterations;
            result.push_back(evaluated[i][best]);
        }
        return result;
    }

private:
    utils::registration::PointCloud measurement_cloud(const typename Base::MeasVector& measurement) const {
        BREW_ASSERT(icp_runner_.ready(), "TmEkf: ICP and template library must be set");
        BREW_ASSERT(measurement.size() > 0 && measurement.size() % 3 == 0,
            "TmEkf: measurement must contain one or more packed 3D points");
        return utils::registration::PointCloud(Eigen::Map<const Eigen::MatrixXd>(
            measurement.data(), 3, measurement.size()/3));
    }

    utils::registration::IcpPseudoMeasurement register_cloud(
            const utils::registration::PointCloud& meas_cloud, const Dist& predicted) const {
        constexpr int d = 3;
        const int n_aug = predicted.aug_dim();
        const auto& pos_idx = predicted.pos_indices();
        // Sibling impulse branches must use the same registration initialization.
        // Remove only the positive branch's nominal impulse increment here.
        Eigen::Vector3d registration_position = predicted.position();
        if (predicted.impulse_branch() == 1
            && predicted.impulse_response().size() == n_aug && impulse_max() > 0.0) {
            for (int i = 0; i < d; ++i) {
                registration_position[i] -= impulse_model()->mean()
                    * predicted.impulse_response()[pos_idx[i]];
            }
        }
        const auto psm = icp_runner_.run(
            meas_cloud, predicted.template_id(),
            predicted.rotation(), registration_position, predicted.needs_pca_alignment());
        return psm;

    }

    double surface_score(const utils::registration::PointCloud& cloud, int template_id,
            const utils::registration::IcpPseudoMeasurement& fit) const {
        return std::isfinite(fit.log_likelihood) ? icp_runner_.surface_log_likelihood(
            cloud, template_id, fit.R, fit.t,
            surface_sigma_sq_, surface_outlier_probability_, surface_outlier_volume_)
            : -std::numeric_limits<double>::infinity();
    }

    LogCorrectionResult correct_registered(const Dist& predicted,
            const utils::registration::IcpPseudoMeasurement& psm, double log_surface) const {
        constexpr int d = 3;
        const int n_trans = predicted.trans_dim();
        const int n_rot = predicted.rot_dim();
        const int n_aug = predicted.aug_dim();
        const auto& pos_idx = predicted.pos_indices();
        Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(d, n_trans);
        for (int i = 0; i < d; ++i) {
            H_pos(i, pos_idx[i]) = 1.0;
        }

        constexpr int m_dim = d + 3;
        Eigen::Matrix<double, m_dim, 1> innovation;
        innovation.head<3>() = psm.t - H_pos * predicted.mean();
        innovation.tail<3>() = psm.rotation_innovation();

        Eigen::MatrixXd H_aug = Eigen::MatrixXd::Zero(m_dim, n_aug);
        H_aug.topLeftCorner(d, n_trans) = H_pos;
        H_aug.bottomRightCorner(n_rot, n_rot) = Eigen::Matrix3d::Identity();

        Eigen::MatrixXd R_pose = this->measurement_noise_;
        BREW_ASSERT(R_pose.rows() == m_dim && R_pose.cols() == m_dim,
                    "TmEkf: pose measurement noise must be 6x6");
        R_pose.topRightCorner<3, 3>().setZero();
        R_pose.bottomLeftCorner<3, 3>().setZero();
        R_pose.bottomRightCorner<3, 3>() = icp_runner_.pose_rotation_covariance(
            predicted.template_id(), psm.R, psm.t.norm(),
            pose_point_sigma_sq_, pose_angular_pitch_, pose_return_cap_);

        Eigen::MatrixXd P = predicted.covariance();
        Eigen::VectorXd response = Eigen::VectorXd::Zero(n_aug);
        const bool impulse_enabled = predicted.impulse_branch() == 1
            && predicted.impulse_response().size() == n_aug
            && predicted.impulse_response().squaredNorm() > 0.0 && impulse_max() > 0.0;
        const double prior_mean = impulse_enabled ? impulse_model()->mean() : 0.0;
        const double prior_variance = impulse_enabled ? impulse_model()->variance() : 0.0;
        if (impulse_enabled) {
            response = predicted.impulse_response();
            P -= prior_variance * response * response.transpose();
        }
        const Eigen::VectorXd measurement_response = H_aug * response;
        // Condition the state on the latent magnitude, then marginalize its posterior.
        const Eigen::VectorXd residual = innovation + prior_mean * measurement_response;
        Eigen::MatrixXd S = H_aug * P * H_aug.transpose() + R_pose;
        S = 0.5 * (S + S.transpose());
        auto S_ldlt = S.ldlt();
        Eigen::MatrixXd K = P * H_aug.transpose()
            * S_ldlt.solve(Eigen::MatrixXd::Identity(m_dim, m_dim));
        double observation = std::numeric_limits<double>::quiet_NaN();
        double observation_variance = std::numeric_limits<double>::quiet_NaN();
        double posterior_mean = 0.0, posterior_variance = 0.0;
        double log_magnitude_mass = 0.0;
        if (impulse_enabled) {
            const Eigen::VectorXd whitened = S_ldlt.solve(measurement_response);
            observation_variance = 1.0 / measurement_response.dot(whitened);
            observation = observation_variance * whitened.dot(residual);
            std::tie(posterior_mean, posterior_variance, log_magnitude_mass) = impulse_posterior_moments(
                observation, observation_variance);
        }
        const Eigen::VectorXd posterior_response = response - K * measurement_response;
        const Eigen::VectorXd delta = K * residual - prior_mean * response
            + posterior_mean * posterior_response;
        Eigen::VectorXd next_mean = predicted.mean() + delta.head(n_trans);
        Eigen::Matrix3d next_rotation = psm.apply_rotation_delta(delta.tail<3>());
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n_aug, n_aug) - K * H_aug;
        Eigen::MatrixXd next_cov = A * P * A.transpose() + K * R_pose * K.transpose()
            + posterior_variance * posterior_response * posterior_response.transpose();
        // Recenter the left attitude error about the corrected nominal rotation.
        Eigen::MatrixXd reset = Eigen::MatrixXd::Identity(n_aug, n_aug);
        reset.bottomRightCorner<3, 3>() = utils::registration::so3::left_jacobian(delta.tail<3>());
        next_cov = reset * next_cov * reset.transpose();
        next_cov = 0.5 * (next_cov + next_cov.transpose());
        auto corrected = predicted.with_updated_state_aligned(
            std::move(next_mean), std::move(next_cov), std::move(next_rotation));
        corrected.set_impulse_evidence(observation, observation_variance);
        double log_pose_evidence = -0.5 * (residual.dot(S_ldlt.solve(residual))
            + S_ldlt.vectorD().array().log().sum() + m_dim * std::log(2.0 * M_PI));
        if (impulse_enabled) {
            log_pose_evidence += 0.5 * observation * observation / observation_variance
                + log_magnitude_mass;
        }
        // Integrate a local Gaussian approximation of the surface likelihood
        // against the predicted pose. Its curvature volume makes the integral
        // approach the surface peak as prior uncertainty tends to zero.
        const double log_pose_volume = 0.5 * (m_dim * std::log(2.0 * M_PI)
            + R_pose.ldlt().vectorD().array().log().sum());
        const double log_L = log_surface + log_pose_evidence + log_pose_volume;
        corrected.set_impulse_log_likelihood(log_L);
        const double likelihood = std::isfinite(log_L)
            ? std::exp(std::clamp(log_L, -700.0, 700.0)) : 0.0;

        update_firing_report(corrected);
        return {{std::move(corrected), likelihood}, log_L};
    }

public:
    [[nodiscard]] double gate(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {

        constexpr int d = 3;
        const auto& pos_idx = predicted.pos_indices();

        const int M = static_cast<int>(measurement.size()) / d;
        BREW_ASSERT(M > 0 && measurement.size() % d == 0,
            "TmEkf: measurement must contain one or more packed 3D points");
        Eigen::MatrixXd meas_mat = Eigen::Map<const Eigen::MatrixXd>(measurement.data(), d, M);
        Eigen::Vector3d centroid = meas_mat.rowwise().mean();

        Eigen::Vector3d t_pred = predicted.position();

        Eigen::Matrix3d P_pos;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                P_pos(i, j) = predicted.covariance()(pos_idx[i], pos_idx[j]);
            }
        }

        Eigen::Matrix3d R_pos = Eigen::Matrix3d::Zero();
        if (this->measurement_noise_.rows() >= d && this->measurement_noise_.cols() >= d) {
            R_pos = this->measurement_noise_.topLeftCorner(d, d);
        }
        // Geometric spread makes the gate tolerant of a displaced visible centroid.
        // It is not added to the registered-pose measurement covariance.
        const Eigen::Matrix3d S = 0.5 * (P_pos + P_pos.transpose()) + R_pos
            + icp_runner_.template_spread(predicted.template_id(), predicted.rotation());
        Eigen::Vector3d nu = centroid - t_pred;
        return nu.transpose() * S.ldlt().solve(nu);
    }

    void set_rotation_process_noise(Eigen::MatrixXd Q_rot) {
        rotation_process_noise_ = std::move(Q_rot);
    }

    void set_surface_sigma_sq(double value) {
        BREW_ASSERT(std::isfinite(value) && value > 0.0,
                    "TmEkf: surface sigma squared must be positive");
        surface_sigma_sq_ = value;
    }

    void set_surface_outlier_probability(double value) {
        BREW_ASSERT(std::isfinite(value) && value >= 0.0 && value < 1.0,
                    "TmEkf: surface outlier probability must be in [0, 1)");
        surface_outlier_probability_ = value;
    }

    void set_surface_outlier_volume(double value) {
        BREW_ASSERT(std::isfinite(value) && value > 0.0,
                    "TmEkf: surface outlier volume must be positive");
        surface_outlier_volume_ = value;
    }

    void set_pose_point_sigma_sq(double value) {
        BREW_ASSERT(std::isfinite(value) && value > 0.0,
                    "TmEkf: pose point variance must be positive");
        pose_point_sigma_sq_ = value;
    }

    void set_pose_angular_pitch(double value) {
        BREW_ASSERT(std::isfinite(value) && value > 0.0,
                    "TmEkf: pose angular pitch must be positive");
        pose_angular_pitch_ = value;
    }

    void set_pose_return_cap(int value) {
        BREW_ASSERT(value > 0, "TmEkf: pose return cap must be positive");
        pose_return_cap_ = value;
    }

    // Report inference uses the same physical prior as the primary recursion.
    // It does not modify state moments, measurement likelihoods, or RFS weights.
    void update_firing_report(Dist& current) const {
        if (impulse_max() <= 0.0 || current.impulse_parent() < 0) { current.set_firing({}); return; }
        using namespace utils::firing_detail;
        auto report = firing_prior(impulse_probability(), impulse_max());
        if (std::isfinite(current.impulse_observation())) {
            const auto branch = std::isfinite(current.impulse_log_zero()) && std::isfinite(current.impulse_log_positive())
                ? std::array<double, 2>{current.impulse_log_zero(), current.impulse_log_positive()}
                : impulse_log_evidence(current.impulse_observation(), current.impulse_variance(), impulse_max());
            const double burn = std::log(impulse_probability()) + branch[1];
            const double probability = std::exp(burn - log_add(
                std::log1p(-impulse_probability()) + branch[0], burn));
            report = impulse_posterior(current.impulse_observation(), current.impulse_variance(),
                                       probability, impulse_max());
        }
        current.set_firing(std::move(report));
    }

    void set_icp(std::shared_ptr<utils::registration::IcpBase> icp) {
        icp_runner_.set_icp(std::move(icp));
    }

    void set_template_library(std::shared_ptr<utils::registration::TemplateLibrary> lib) {
        icp_runner_.set_template_library(std::move(lib));
    }

    [[nodiscard]] int last_icp_iterations() const { return last_icp_iterations_; }

protected:
    const dynamics::ImpulseModel* impulse_model() const {
        return this->dyn_obj_ ? this->dyn_obj_->impulse_model() : nullptr;
    }
    double impulse_max() const { return impulse_model() ? impulse_model()->prior().maximum : 0.; }
    double impulse_probability() const { return impulse_model() ? impulse_model()->prior().probability : 0.; }

    [[nodiscard]] std::tuple<double, double, double> impulse_posterior_moments(
        double observation, double variance) const {
        // Closed-form truncated-normal moments and uniform-prior evidence.
        // Extended precision protects the variance from cancellation near a bound.
        const long double log_sqrt_two_pi = 0.5L * std::log(2.0L * std::acos(-1.0L));
        const auto log_cdf = [log_sqrt_two_pi](long double x) {
            if (x >= -20.0L) {
                return x <= 0.0L ? std::log(0.5L * std::erfc(-x / std::sqrt(2.0L)))
                    : std::log1p(-0.5L * std::erfc(x / std::sqrt(2.0L)));
            }
            // Normal-tail asymptotic expansion evaluates the CDF without underflow.
            long double sum = 1.0L, term = 1.0L;
            for (int n = 1; ; ++n) {
                const long double next = -term * (2 * n - 1) / (x * x);
                if (std::abs(next) >= std::abs(term)) break;
                sum += next;
                if (std::abs(next) <= std::numeric_limits<long double>::epsilon() * sum) break;
                term = next;
            }
            return -0.5L * x * x - std::log(-x) - log_sqrt_two_pi + std::log(sum);
        };
        const long double sigma = std::sqrt(static_cast<long double>(variance));
        const long double lower = -static_cast<long double>(observation) / sigma;
        const long double upper = (static_cast<long double>(impulse_max()) - observation) / sigma;
        // Use survival probabilities for a positive interval and expm1 for close bounds.
        const long double large = lower > 0.0L ? log_cdf(-lower) : log_cdf(upper);
        const long double small = lower > 0.0L ? log_cdf(-upper) : log_cdf(lower);
        const long double log_mass = large + std::log(-std::expm1(small - large));
        const long double lower_ratio = std::exp(-0.5L * lower * lower - log_sqrt_two_pi - log_mass);
        const long double upper_ratio = std::exp(-0.5L * upper * upper - log_sqrt_two_pi - log_mass);
        const long double shift = lower_ratio - upper_ratio;
        const long double mean = observation + sigma * shift;
        const long double posterior_variance = variance
            * (1.0L + lower * lower_ratio - upper * upper_ratio - shift * shift);
        // The caller supplies the remaining multivariate Gaussian likelihood factor.
        const long double log_evidence = log_mass + std::log(sigma)
            + log_sqrt_two_pi - std::log(static_cast<long double>(impulse_max()));
        return {static_cast<double>(mean), static_cast<double>(std::max(0.0L, posterior_variance)),
                static_cast<double>(log_evidence)};
    }

    [[nodiscard]] Dist predict_with(
        double ,
        const Dist& prev,
        Eigen::MatrixXd F_aug,
        Eigen::VectorXd next_mean,
        Eigen::Matrix3d next_rotation,
        const Eigen::MatrixXd& process_noise) const {

        const int n_trans = prev.trans_dim();
        const int n_rot = prev.rot_dim();
        const int n_aug = prev.aug_dim();

        Eigen::MatrixXd Q_aug = Eigen::MatrixXd::Zero(n_aug, n_aug);
        Q_aug.topLeftCorner(n_trans, n_trans) = process_noise;
        Q_aug.bottomRightCorner(n_rot, n_rot) = rotation_process_noise_;

        Eigen::MatrixXd next_cov = F_aug * prev.covariance() * F_aug.transpose() + Q_aug;
        next_cov = 0.5 * (next_cov + next_cov.transpose());

        auto out = prev.with_updated_state(
            std::move(next_mean), std::move(next_cov), std::move(next_rotation));
        out.set_mode_id(prev.mode_id());
        return out;
    }

    Eigen::MatrixXd rotation_process_noise_;
    double surface_sigma_sq_ = 0.25;
    double surface_outlier_probability_ = 0.05;
    double surface_outlier_volume_ = 16.0 * 16.0 * 16.0;
    double pose_point_sigma_sq_ = 0.25 * 0.25;
    double pose_angular_pitch_ = (20.0 * M_PI / 180.0) / 256.0;
    int pose_return_cap_ = 5000;
    utils::registration::TmIcpRunner icp_runner_;
    mutable int last_icp_iterations_ = 0;
};

}
