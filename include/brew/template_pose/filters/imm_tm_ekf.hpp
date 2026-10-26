#pragma once

#include "brew/template_pose/filters/tm_ekf.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace brew::filters {

// @mex filter
// @mex_name ImmTmEkf
// @mex_dist TemplatePose
// @mex_setters rotation_process_noise:mat, surface_sigma_sq:double, surface_outlier_probability:double, surface_outlier_volume:double, pose_point_sigma_sq:double, pose_angular_pitch:double, pose_return_cap:int, transition_matrix:mat, maneuver_accel_sigma:double, thruster_axis:vec, intent_model:double, intent_attitude_score:double, nominal_accel:double, intent_gain_mean:double, intent_gain_std:double, intent_attitude_sigma:double
// @mex_handle_setters icp:IcpBase, template_library:TemplateLibrary

template <typename Scalar = double, int D = Eigen::Dynamic>
class ImmTmEkf : public TmEkf<Scalar, D> {
public:
    using Dist = models::TemplatePose<Scalar, D>;
    using Base = Filter<Dist>;
    using TmBase = TmEkf<Scalar, D>;
    using typename Base::DynamicsType;

    ImmTmEkf() {
        transition_matrix_.resize(2, 2);
        transition_matrix_ << 0.93, 0.07,
                              0.18, 0.82;
        thruster_axis_.resize(3);
        thruster_axis_ << 1.0, 0.0, 0.0;
    }

    [[nodiscard]] std::unique_ptr<Base> clone() const override {
        auto c = std::make_unique<ImmTmEkf<Scalar, D>>();
        c->dyn_obj_ = this->dyn_obj_;
        c->h_ = this->h_;
        c->H_ = this->H_;
        c->process_noise_ = this->process_noise_;
        c->measurement_noise_ = this->measurement_noise_;
        c->rotation_process_noise_ = this->rotation_process_noise_;
        c->surface_sigma_sq_ = this->surface_sigma_sq_;
        c->surface_outlier_probability_ = this->surface_outlier_probability_;
        c->surface_outlier_volume_ = this->surface_outlier_volume_;
        c->pose_point_sigma_sq_ = this->pose_point_sigma_sq_;
        c->pose_angular_pitch_ = this->pose_angular_pitch_;
        c->pose_return_cap_ = this->pose_return_cap_;
        c->icp_runner_ = this->icp_runner_;
        c->transition_matrix_ = transition_matrix_;
        c->maneuver_accel_sigma_ = maneuver_accel_sigma_;
        c->thruster_axis_ = thruster_axis_;
        c->intent_model_ = intent_model_;
        c->intent_attitude_score_ = intent_attitude_score_;
        c->nominal_accel_ = nominal_accel_;
        c->intent_gain_mean_ = intent_gain_mean_;
        c->intent_gain_std_ = intent_gain_std_;
        c->intent_attitude_sigma_ = intent_attitude_sigma_;
        return c;
    }

    void set_transition_matrix(Eigen::MatrixXd Pi) {
        transition_matrix_ = std::move(Pi);
    }

    void set_maneuver_accel_sigma(double sigma) {
        maneuver_accel_sigma_ = sigma;
    }

    void set_thruster_axis(Eigen::VectorXd axis) {
        thruster_axis_ = std::move(axis);
        const double n = thruster_axis_.norm();
        if (n > 0.0) thruster_axis_ /= n;
    }

    void set_intent_model(double enabled) { intent_model_ = enabled > 0.5; }
    void set_intent_attitude_score(double enabled) { intent_attitude_score_ = enabled > 0.5; }
    void set_nominal_accel(double a) { nominal_accel_ = a; }
    void set_intent_gain_mean(double mean) {
        BREW_ASSERT(std::isfinite(mean), "ImmTmEkf: gain mean must be finite");
        intent_gain_mean_ = mean;
    }
    void set_intent_gain_std(double std) {
        BREW_ASSERT(std::isfinite(std) && std >= 0.0,
                    "ImmTmEkf: gain standard deviation must be nonnegative");
        intent_gain_std_ = std;
    }
    void set_intent_attitude_sigma(double sigma) {
        BREW_ASSERT(std::isfinite(sigma) && sigma > 0.0,
                    "ImmTmEkf: attitude policy standard deviation must be positive");
        intent_attitude_sigma_ = sigma;
    }

    [[nodiscard]] static double integrated_gain_log_score(
        const Eigen::Vector3d& increment, const Eigen::Vector3d& error,
        const Eigen::Matrix3d& pose_covariance, double policy_sigma,
        double gain_mean, double gain_std) {
        Eigen::Matrix3d covariance = pose_covariance;
        covariance.diagonal().array() += policy_sigma * policy_sigma;
        covariance.noalias() += gain_std * gain_std * (error * error.transpose());
        covariance = 0.5 * (covariance + covariance.transpose());
        Eigen::LDLT<Eigen::Matrix3d> ldlt(covariance);
        BREW_ASSERT(ldlt.info() == Eigen::Success &&
                    (ldlt.vectorD().array() > 0.0).all(),
                    "ImmTmEkf: attitude score covariance must be positive definite");
        const Eigen::Vector3d residual = increment - gain_mean * error;
        return -0.5 * (residual.dot(ldlt.solve(residual))
            + ldlt.vectorD().array().log().sum()
            + 3.0 * std::log(2.0 * std::numbers::pi));
    }

    [[nodiscard]] Dist predict(double dt, const Dist& prev) const override {
        Dist out = predict_mode(dt, prev, std::clamp(prev.mode_id(), 0, mode_count() - 1));
        return out;
    }

    void predict_batch_dynamic(double dt, models::Mixture<Dist, Eigen::Dynamic>& mix) const override {
        if (intent_model_) {
            predict_intent_batch(dt, mix);
            return;
        }
        models::Mixture<Dist, Eigen::Dynamic> expanded;
        const int modes = mode_count();
        for (std::size_t i = 0; i < mix.size(); ++i) {
            const int prev_mode = std::clamp(mix.component(i).mode_id(), 0, modes - 1);
            for (int next_mode = 0; next_mode < modes; ++next_mode) {
                const double pij = transition_matrix_(prev_mode, next_mode);
                if (pij <= 0.0) continue;
                auto pred = predict_mode(dt, mix.component(i), next_mode);
                expanded.add_component(std::make_unique<Dist>(std::move(pred)), mix.weight(i) * pij);
            }
        }
        mix = std::move(expanded);
    }

    [[nodiscard]] typename Base::CorrectionResult correct(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {
        return correct_with_log(measurement, predicted).first;
    }

    [[nodiscard]] typename Base::LogCorrectionResult correct_with_log(
        const typename Base::MeasVector& measurement,
        const Dist& predicted) const override {
        auto result = TmBase::correct_with_log(measurement, predicted);
        if (!intent_model_ || !intent_attitude_score_ || predicted.mode_id() < 0 || predicted.mode_id() >= 5
            || predicted.needs_pca_alignment()) {
            return result;
        }
        const Eigen::Matrix3d command = expected_rotation(predicted.mode_id(), result.first.distribution.mean());
        const Eigen::Vector3d increment = log_so3(result.first.distribution.rotation() * predicted.rotation().transpose());
        const Eigen::Vector3d error = log_so3(command * predicted.rotation().transpose());
        result.second += integrated_gain_log_score(
            increment, error,
            result.first.distribution.covariance().template bottomRightCorner<3, 3>(),
            intent_attitude_sigma_, intent_gain_mean_, intent_gain_std_);
        result.first.likelihood = std::isfinite(result.second)
            ? std::exp(std::clamp(result.second, -700.0, 700.0)) : 0.0;
        return result;
    }

private:
    void predict_intent_batch(double dt, models::Mixture<Dist, Eigen::Dynamic>& mix) const {
        models::Mixture<Dist, Eigen::Dynamic> predicted;
        std::vector<bool> used(mix.size(), false);
        for (std::size_t seed = 0; seed < mix.size(); ++seed) {
            if (used[seed]) continue;
            std::vector<std::size_t> group;
            for (std::size_t index = seed; index < mix.size(); ++index) {
                if (used[index]) continue;
                if (mix.component(index).template_id() == mix.component(seed).template_id()
                    && (mix.component(index).position() - mix.component(seed).position()).norm() < 20.0) {
                    used[index] = true;
                    group.push_back(index);
                }
            }
            for (int next_mode = 0; next_mode < mode_count(); ++next_mode) {
                double total = 0.0;
                for (auto index : group) {
                    const int source = mix.component(index).mode_id();
                    const double probability = source < 0
                        ? 1.0 / mode_count()
                        : transition_matrix_(std::clamp(source, 0, mode_count() - 1), next_mode);
                    total += mix.weight(index) * probability;
                }
                if (total <= 0.0) continue;
                Eigen::VectorXd mean = Eigen::VectorXd::Zero(mix.component(seed).trans_dim());
                Eigen::Matrix3d rotation_sum = Eigen::Matrix3d::Zero();
                for (auto index : group) {
                    const int source = mix.component(index).mode_id();
                    const double probability = source < 0
                        ? 1.0 / mode_count()
                        : transition_matrix_(std::clamp(source, 0, mode_count() - 1), next_mode);
                    const double weight = mix.weight(index) * probability / total;
                    mean += weight * mix.component(index).mean();
                    rotation_sum += weight * mix.component(index).rotation();
                }
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                    rotation_sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
                correction(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
                Eigen::Matrix3d rotation = svd.matrixU() * correction * svd.matrixV().transpose();
                Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(
                    mix.component(seed).aug_dim(), mix.component(seed).aug_dim());
                std::size_t base = seed;
                for (auto index : group) {
                    const int source = mix.component(index).mode_id();
                    const double probability = source < 0
                        ? 1.0 / mode_count()
                        : transition_matrix_(std::clamp(source, 0, mode_count() - 1), next_mode);
                    const double weight = mix.weight(index) * probability / total;
                    const Eigen::VectorXd difference = mix.component(index).mean() - mean;
                    Eigen::MatrixXd spread = Eigen::MatrixXd::Zero(covariance.rows(), covariance.cols());
                    spread.topLeftCorner(difference.size(), difference.size()) = difference * difference.transpose();
                    covariance += weight * (mix.component(index).covariance() + spread);
                    if (!mix.component(index).needs_pca_alignment()) base = index;
                }
                Dist mixed = mix.component(base).with_updated_state(mean, covariance, rotation);
                mixed.set_mode_id(next_mode);
                auto next = predict_mode(dt, mixed, next_mode);
                predicted.add_component(std::make_unique<Dist>(std::move(next)), total);
            }
        }
        mix = std::move(predicted);
    }

    [[nodiscard]] int mode_count() const {
        return static_cast<int>(transition_matrix_.rows());
    }

    [[nodiscard]] Dist predict_mode(double dt, const Dist& prev, int mode) const {
        const int n_trans = prev.trans_dim();
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
                : template_matching::so3::exp(dt * Eigen::Vector3d(prev.mean().template segment<3>(6)))
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
            if (!prev.needs_pca_alignment())
                F_aug.block<3, 3>(n_trans, 6) = dt * Eigen::Matrix3d::Identity();
        } else {
            F_aug.topLeftCorner(n_trans, n_trans) = F;
        }

        Eigen::MatrixXd Q = this->process_noise_;
        if (mode > 0 && n_trans >= 6 && thruster_axis_.size() == 3) {
            Eigen::Vector3d b = thruster_axis_.template head<3>();
            if (b.norm() > 0.0) b.normalize();
            Eigen::Vector3d d = prev.rotation() * b;
            Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(n_trans, 3);
            Gamma.block(0, 0, 3, 3) = 0.5 * dt * dt * Eigen::Matrix3d::Identity();
            Gamma.block(3, 0, 3, 3) = dt * Eigen::Matrix3d::Identity();
            Q += Gamma * (maneuver_accel_sigma_ * maneuver_accel_sigma_ * (d * d.transpose())) * Gamma.transpose();
        }

        if (intent_model_ && n_trans >= 6) {
            const Eigen::Vector3d u = policy_control(mode, prev.mean());
            next_mean.head(3) += 0.5 * dt * dt * u;
            next_mean.segment(3, 3) += dt * u;
        }

        Dist out = this->predict_with(dt, prev, std::move(F_aug), std::move(next_mean),
                                      std::move(next_rotation), Q);
        out.set_mode_id(mode);
        return out;
    }

    [[nodiscard]] Eigen::Vector3d policy_control(int mode, const Eigen::VectorXd& state) const {
        Eigen::Vector3d r = state.head<3>();
        const double rn = std::max(r.norm(), 1e-12);
        const Eigen::Vector3d rhat = r / rn;
        if (mode == 1) return -nominal_accel_ * rhat;
        if (mode == 2) return nominal_accel_ * rhat;
        if (mode == 3) return nominal_accel_ * Eigen::Vector3d::UnitY();
        if (mode == 4) {
            Eigen::Vector3d tangent = Eigen::Vector3d::UnitZ().cross(rhat);
            return nominal_accel_ * tangent / std::max(tangent.norm(), 1e-12);
        }
        return Eigen::Vector3d::Zero();
    }

    [[nodiscard]] static Eigen::Matrix3d expected_rotation(int mode, const Eigen::VectorXd& state) {
        Eigen::Vector3d rhat = state.head<3>() / std::max(state.head<3>().norm(), 1e-12);
        Eigen::Vector3d x = Eigen::Vector3d::UnitX();
        if (mode == 1) x = -rhat;
        else if (mode == 2) x = rhat;
        else if (mode == 3) x = Eigen::Vector3d::UnitY();
        else if (mode == 4) {
            x = Eigen::Vector3d::UnitZ().cross(rhat);
            x /= std::max(x.norm(), 1e-12);
        }
        Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
        if (mode == 4) z = -rhat;
        z -= x * x.dot(z);
        if (z.norm() < 1e-8) {
            z = Eigen::Vector3d::UnitY() - x * x.dot(Eigen::Vector3d::UnitY());
        }
        z.normalize();
        Eigen::Vector3d y = z.cross(x);
        Eigen::Matrix3d R;
        R.col(0) = x; R.col(1) = y; R.col(2) = z;
        return R;
    }

    [[nodiscard]] static Eigen::Vector3d log_so3(const Eigen::Matrix3d& R) {
        const double cosine = std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
        const double angle = std::acos(cosine);
        Eigen::Vector3d vee(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        if (angle < 1e-8) return 0.5 * vee;
        return angle / (2.0 * std::sin(angle)) * vee;
    }

    Eigen::MatrixXd transition_matrix_;
    double maneuver_accel_sigma_ = 0.03;
    Eigen::VectorXd thruster_axis_;
    bool intent_model_ = false;
    bool intent_attitude_score_ = true;
    double nominal_accel_ = 0.0;
    double intent_gain_mean_ = 0.125;
    double intent_gain_std_ = 0.25 / std::sqrt(12.0);
    double intent_attitude_sigma_ = 0.08726646259971647;
};

template <typename Scalar, int D>
struct default_filter<models::TemplatePose<Scalar, D>> { using type = ImmTmEkf<Scalar, D>; };

}
