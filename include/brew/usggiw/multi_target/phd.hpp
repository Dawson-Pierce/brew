#pragma once

#include "brew/clustering/dbscan.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/rfs_base.hpp"
#include "brew/usggiw/merge.hpp"
#include <cmath>
#include <deque>

namespace brew::usggiw {

// @mex rfs
// @mex_name PHD
// @mex_params prune_threshold:double:1e-4, merge_threshold:double:4.0, max_components:int:100, extract_threshold:double:0.5, gate_threshold:double:9.0
// @mex_init set_intensity
// @mex_has birth_weights, cluster_object
template<typename Scalar = double, int De = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class PHD : public multi_target::RFSBase {
    using T = models::USGGIW<Scalar, De>;
    using Mixture = models::Mixture<T, MaxComponents>;
public:
    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto out = std::make_unique<PHD>();
        out->prob_detection_ = prob_detection_; out->prob_survive_ = prob_survive_;
        out->clutter_rate_ = clutter_rate_; out->clutter_density_ = clutter_density_;
        if (filter_) out->filter_ = filter_->clone();
        if (intensity_) out->intensity_ = intensity_->clone();
        if (birth_model_) out->birth_model_ = birth_model_->clone();
        out->prune_threshold_ = prune_threshold_; out->merge_threshold_ = merge_threshold_;
        out->max_components_ = max_components_; out->extract_threshold_ = extract_threshold_;
        out->gate_threshold_ = gate_threshold_; out->cluster_obj_ = cluster_obj_;
        out->detect_field_ = detect_field_; out->detect_radius_ = detect_radius_;
        out->prob_detection_out_ = prob_detection_out_;
        return out;
    }

    void set_filter(std::unique_ptr<filters::Filter<T>> value) { filter_ = std::move(value); }
    void set_birth_model(std::unique_ptr<Mixture> value) {
        birth_model_ = std::move(value);
        if (birth_model_ && !birth_model_->empty() && !cluster_obj_)
            cluster_obj_ = std::make_shared<clustering::DBSCAN>();
    }
    void set_intensity(std::unique_ptr<Mixture> value) { intensity_ = std::move(value); }
    void set_prune_threshold(double value) { prune_threshold_ = value; }
    void set_merge_threshold(double value) { merge_threshold_ = value; }
    void set_max_components(int value) { max_components_ = value; }
    void set_extract_threshold(double value) { extract_threshold_ = value; }
    void set_gate_threshold(double value) { gate_threshold_ = value; }
    void set_cluster_object(std::shared_ptr<clustering::ClusterBase> value) { cluster_obj_ = std::move(value); }
    void set_birth_weights(const Eigen::VectorXd& weights) {
        if (!birth_model_) return;
        if (weights.size() == 1) birth_model_->weights().setConstant(weights(0));
        else birth_model_->weights() = weights;
    }
    void set_detection_field(const Eigen::MatrixXd& field, double radius, double pd_out) {
        detect_field_ = field; detect_radius_ = radius; prob_detection_out_ = pd_out;
    }
    void set_measurement_components(const Mixture& value) { measurement_components_ = value.clone(); }
    [[nodiscard]] const Mixture& intensity() const { return *intensity_; }
    [[nodiscard]] Mixture& intensity() { return *intensity_; }
    [[nodiscard]] const std::deque<std::unique_ptr<Mixture>>& extracted_mixtures() const { return extracted_; }

    void predict(int, double dt) override {
        if (!intensity_ || !filter_) return;
        for (std::size_t i = 0; i < intensity_->size(); ++i) intensity_->weights()(i) *= prob_survive_;
        filter_->predict_batch_dynamic(dt, *intensity_);
        if (birth_model_) intensity_->add_components(*birth_model_->clone());
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !filter_) return;
        auto* sphere_filter = dynamic_cast<filters::USGGIWEKF<Scalar, De>*>(filter_.get());
        const bool component_mode = sphere_filter && measurement_components_
            && measurement_components_->size() == static_cast<std::size_t>(measurements.cols());
        std::vector<Eigen::MatrixXd> groups = (!component_mode && cluster_obj_)
            ? cluster_obj_->cluster(measurements) : columns(measurements);
        Mixture missed;
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            const double pd = pd_at(intensity_->component(i).direction().template cast<double>());
            const double prior_weight = intensity_->weight(i);
            if (prior_weight * (1.0 - pd) > 0.0)
                missed.add_component(intensity_->component(i).clone_typed(),
                                     prior_weight * (1.0 - pd));
            const double zero_count_probability = std::pow(
                intensity_->component(i).beta()
                    / (intensity_->component(i).beta() + 1.0),
                intensity_->component(i).alpha());
            if (prior_weight * pd * zero_count_probability > 0.0) {
                auto zero_count = intensity_->component(i).clone_typed();
                zero_count->beta() += 1.0;
                missed.add_component(std::move(zero_count),
                                     prior_weight * pd * zero_count_probability);
            }
            intensity_->weights()(i) *= pd;
        }
        Mixture detected;
        for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
            const auto& group = groups[group_index];
            Eigen::VectorXd packed(group.size());
            for (int j = 0; j < group.cols(); ++j)
                packed.segment(j * group.rows(), group.rows()) = group.col(j);
            const Eigen::Vector3d gate_measurement = normalized(group.rowwise().sum());
            std::vector<double> weights;
            std::vector<std::unique_ptr<T>> distributions;
            for (std::size_t i = 0; i < intensity_->size(); ++i) {
                const double distance = component_mode
                    ? sphere_filter->gate_component(measurement_components_->component(group_index),
                                                    intensity_->component(i))
                    : filter_->gate(gate_measurement, intensity_->component(i));
                if (distance >= gate_threshold_) continue;
                auto result = component_mode
                    ? sphere_filter->correct_component(measurement_components_->component(group_index),
                                                       intensity_->component(i))
                    : filter_->correct(packed, intensity_->component(i));
                weights.push_back(result.likelihood * intensity_->weight(i));
                distributions.push_back(result.distribution.clone_typed());
            }
            double sum = 0.0;
            for (double w : weights) sum += w;
            const double clutter_intensity = std::max(
                clutter_rate_ * clutter_density_, 1e-300);
            const double clutter = std::pow(
                clutter_intensity, component_mode ? 1 : group.cols());
            for (std::size_t i = 0; i < weights.size(); ++i)
                detected.add_component(std::move(distributions[i]), weights[i] / (clutter + sum));
        }
        *intensity_ = std::move(missed);
        intensity_->add_components(detected);
        measurement_components_.reset();
    }

    void cleanup() override {
        if (!intensity_) return;
        fusion::prune(*intensity_, prune_threshold_);
        merge(*intensity_, merge_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));
        this->push_history(extracted_, extract());
    }

    [[nodiscard]] std::unique_ptr<Mixture> extract() const {
        auto out = std::make_unique<Mixture>();
        if (!intensity_) return out;
        for (std::size_t i = 0; i < intensity_->size(); ++i)
            if (intensity_->weight(i) >= extract_threshold_)
                out->add_component(intensity_->component(i).clone_typed(), intensity_->weight(i));
        return out;
    }

private:
    static std::vector<Eigen::MatrixXd> columns(const Eigen::MatrixXd& measurements) {
        std::vector<Eigen::MatrixXd> out;
        for (int i = 0; i < measurements.cols(); ++i) out.emplace_back(measurements.col(i));
        return out;
    }
    [[nodiscard]] double pd_at(const Eigen::Vector3d& direction) const {
        if (detect_radius_ <= 0.0 || detect_field_.cols() == 0) return prob_detection_;
        for (int i = 0; i < detect_field_.cols(); ++i) {
            const Eigen::Vector3d center = normalized(detect_field_.col(i));
            if (std::acos(std::clamp(direction.dot(center), -1.0, 1.0)) <= detect_radius_)
                return prob_detection_;
        }
        return prob_detection_out_;
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    std::unique_ptr<Mixture> intensity_, birth_model_;
    std::unique_ptr<Mixture> measurement_components_;
    std::shared_ptr<clustering::ClusterBase> cluster_obj_;
    Eigen::MatrixXd detect_field_;
    double detect_radius_ = -1.0, prob_detection_out_ = 0.0;
    double prune_threshold_ = 1e-4, merge_threshold_ = 4.0;
    int max_components_ = 100;
    double extract_threshold_ = 0.5, gate_threshold_ = 9.0;
    std::deque<std::unique_ptr<Mixture>> extracted_;
};

}  // namespace brew::usggiw
