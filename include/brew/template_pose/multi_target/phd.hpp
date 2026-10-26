#pragma once

// PHD RFS for the template_pose package (concrete; brew::template_pose).

#include "brew/template_pose/template_pose_model.hpp"

#include "brew/shared/rfs_base.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/trajectory_window.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/filter_traits.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/template_pose/merge.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <vector>

namespace brew::template_pose {

// @mex rfs
// @mex_name PHD
// @mex_params prune_threshold:double:1e-4, merge_threshold:double:4.0, merge_rotation_threshold:double:1.5707963267948966, max_components:int:100, extract_threshold:double:0.5, gate_threshold:double:9.0
// @mex_init set_intensity
// @mex_has cluster_object
template <typename Scalar = double, int D = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class PHD : public multi_target::RFSBase {
    using T = models::TemplatePose<Scalar, D>;
public:
    PHD() = default;

    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto c = std::make_unique<PHD<Scalar, D, MaxComponents>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        c->has_filter_ = has_filter_;
        if (intensity_) c->intensity_ = intensity_->clone();
        if (birth_model_) c->birth_model_ = birth_model_->clone();
        c->prune_threshold_ = prune_threshold_;
        c->merge_threshold_ = merge_threshold_;
        c->merge_rotation_threshold_ = merge_rotation_threshold_;
        c->max_components_ = max_components_;
        c->extract_threshold_ = extract_threshold_;
        c->gate_threshold_ = gate_threshold_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        return c;
    }

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
        has_filter_ = true;
    }

    void set_birth_model(std::unique_ptr<models::Mixture<T, MaxComponents>> birth) {
        if (birth && !birth->empty()) {
            is_extended_ = birth->component(0).is_extended();
            if (is_extended_ && !cluster_obj_) {
                cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
        birth_model_ = std::move(birth);
    }

    void set_intensity(std::unique_ptr<models::Mixture<T, MaxComponents>> intensity) {
        intensity_ = std::move(intensity);
    }

    void set_prune_threshold(double t) { prune_threshold_ = t; }
    void set_merge_threshold(double t) { merge_threshold_ = t; }
    void set_merge_rotation_threshold(double t) { merge_rotation_threshold_ = t; }
    void set_max_components(int n) { max_components_ = n; }
    void set_birth_weights(const Eigen::VectorXd& weights) {
        if (!birth_model_) return;
        if (weights.size() == 1) {
            birth_model_->weights().setConstant(weights(0));
        } else {
            birth_model_->weights() = weights;
        }
    }

    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::ClusterBase> obj) { cluster_obj_ = std::move(obj); }

    [[nodiscard]] const models::Mixture<T, MaxComponents>& intensity() const { return *intensity_; }
    [[nodiscard]] models::Mixture<T, MaxComponents>& intensity() { return *intensity_; }

    [[nodiscard]] const std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    void predict(int , double dt) override {
        if (!intensity_ || !has_filter_) return;

        for (std::size_t k = 0; k < intensity_->size(); ++k) {
            intensity_->weights()(static_cast<Eigen::Index>(k)) *= prob_survive_;
        }
        filter_->predict_batch_dynamic(dt, *intensity_);

        if (birth_model_) {
            auto birth_copy = birth_model_->clone();
            intensity_->add_components(*birth_copy);

        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !has_filter_) return;

        std::vector<Eigen::MatrixXd> meas_groups;
        if (is_extended_ && cluster_obj_) {
            meas_groups = cluster_obj_->cluster(measurements);
        } else {

            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }

        auto undetected = intensity_->clone();
        for (std::size_t i = 0; i < undetected->size(); ++i) {
            undetected->weights()(static_cast<Eigen::Index>(i)) *= (1.0 - prob_detection_);
        }

        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            intensity_->weights()(static_cast<Eigen::Index>(i)) *= prob_detection_;
        }

        models::Mixture<T, MaxComponents> corrected_mix;

        for (std::size_t z = 0; z < meas_groups.size(); ++z) {
            const Eigen::MatrixXd& meas = meas_groups[z];
            Eigen::VectorXd z_gate;
            if (is_extended_ && meas.cols() > 1) {
                z_gate = meas.rowwise().mean();
            } else {
                z_gate = meas.col(0);
            }

            Eigen::VectorXd meas_flat;
            if (meas.cols() > 1) {
                meas_flat.resize(meas.size());
                for (int j = 0; j < meas.cols(); ++j) {
                    meas_flat.segment(j * meas.rows(), meas.rows()) = meas.col(j);
                }
            } else {
                meas_flat = meas.col(0);
            }

            std::vector<double> log_w_lst;
            std::vector<double> prior_weights;
            std::vector<std::unique_ptr<T>> d_lst;

            // Correct both siblings when either branch supports this cloud.
            std::vector<bool> accepted(intensity_->size(), false);
            for (std::size_t k = 0; k < intensity_->size(); ++k) {
                if (filter_->gate(z_gate, intensity_->component(k)) >= gate_threshold_) continue;
                accepted[k] = true;
                if (intensity_->component(k).impulse_parent() < 0) continue;
                for (std::size_t j = 0; j < intensity_->size(); ++j) {
                    if (intensity_->component(j).impulse_parent() == intensity_->component(k).impulse_parent())
                        accepted[j] = true;
                }
            }
            for (std::size_t k = 0; k < intensity_->size(); ++k) {
                if (accepted[k]) {
                    auto [correction, log_likelihood] = filter_->correct_with_log(
                        meas_flat, intensity_->component(k));
                    const double weight = intensity_->weight(k);
                    prior_weights.push_back(weight);
                    log_w_lst.push_back(weight > 0.0
                        ? std::log(weight) + log_likelihood
                        : -std::numeric_limits<double>::infinity());
                    d_lst.push_back(correction.distribution.clone_typed());
                }
            }

            // Split each parent's surface-score mass using predictive pose evidence.
            // The Bernoulli prior is already in the two prediction weights.
            for (std::size_t i = 0; i < d_lst.size(); ++i) {
                const int parent = d_lst[i]->impulse_parent();
                if (parent < 0) continue;
                double max_log = -std::numeric_limits<double>::infinity();
                double parent_weight = 0.0;
                double log_zero = -std::numeric_limits<double>::infinity();
                double log_positive = -std::numeric_limits<double>::infinity();
                double observation = std::numeric_limits<double>::quiet_NaN();
                double variance = std::numeric_limits<double>::quiet_NaN();
                for (std::size_t j = 0; j < d_lst.size(); ++j) {
                    if (d_lst[j]->impulse_parent() != parent) continue;
                    const double evidence = d_lst[j]->impulse_log_likelihood();
                    max_log = std::max(max_log, evidence);
                    parent_weight += prior_weights[j];
                    if (d_lst[j]->impulse_branch() == 0) log_zero = evidence;
                    if (d_lst[j]->impulse_branch() == 1) {
                        log_positive = evidence;
                        observation = d_lst[j]->impulse_observation();
                        variance = d_lst[j]->impulse_variance();
                    }
                }
                if (parent_weight <= 0.0 || !std::isfinite(max_log)) continue;
                double evidence_sum = 0.0;
                for (std::size_t j = 0; j < d_lst.size(); ++j) {
                    if (d_lst[j]->impulse_parent() == parent)
                        evidence_sum += prior_weights[j]
                            * std::exp(d_lst[j]->impulse_log_likelihood() - max_log);
                }
                const double log_normalizer = max_log + std::log(evidence_sum / parent_weight);
                log_w_lst[i] += d_lst[i]->impulse_log_likelihood() - log_normalizer;
                d_lst[i]->set_impulse_branch_evidence(log_zero, log_positive);
                d_lst[i]->set_impulse_evidence(observation, variance);
            }

            const int W = static_cast<int>(meas.cols());
            const double clutter_density = clutter_rate_ * clutter_density_;
            const double log_clutter = clutter_density > 0.0
                ? W * std::log(clutter_density)
                : -std::numeric_limits<double>::infinity();
            double log_denom = log_clutter;
            for (double log_w : log_w_lst) {
                if (!std::isfinite(log_w)) continue;
                if (!std::isfinite(log_denom)) {
                    log_denom = log_w;
                } else {
                    const double max_log = std::max(log_denom, log_w);
                    log_denom = max_log + std::log1p(std::exp(-std::abs(log_denom - log_w)));
                }
            }
            if (!std::isfinite(log_denom)) continue;
            for (std::size_t i = 0; i < log_w_lst.size(); ++i) {
                if (std::isfinite(log_w_lst[i])) {
                    corrected_mix.add_component(
                        std::move(d_lst[i]), std::exp(log_w_lst[i] - log_denom));
                }
            }
        }

        *intensity_ = std::move(*undetected);
        intensity_->add_components(corrected_mix);
    }

    void cleanup() override {
        if (!intensity_) return;
        fusion::prune(*intensity_, prune_threshold_);
        merge(*intensity_, merge_threshold_, merge_rotation_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));

        this->push_history(extracted_mixtures_, extract());
    }

    [[nodiscard]] std::unique_ptr<models::Mixture<T, MaxComponents>> extract() const {
        if (!intensity_) return nullptr;
        auto result = std::make_unique<models::Mixture<T, MaxComponents>>();
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            if (intensity_->weight(i) >= extract_threshold_) {
                result->add_component(
                    intensity_->component(i).clone_typed(),
                    intensity_->weight(i));
            }
        }
        return result;
    }

private:
    std::unique_ptr<filters::Filter<T>> filter_;
    bool has_filter_ = false;
    std::unique_ptr<models::Mixture<T, MaxComponents>> intensity_;
    std::unique_ptr<models::Mixture<T, MaxComponents>> birth_model_;
    std::shared_ptr<clustering::ClusterBase> cluster_obj_;
    double prune_threshold_ = 1e-4;
    double merge_threshold_ = 4.0;
    double merge_rotation_threshold_ = M_PI / 2.0;
    int max_components_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    bool is_extended_ = false;
    std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>> extracted_mixtures_;
};

}
