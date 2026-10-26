#pragma once

#include "brew/clustering/dbscan.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/rfs_base.hpp"
#include "brew/trajectory_usggiw/filters/trajectory_usggiw_ekf.hpp"
#include "brew/trajectory_usggiw/merge.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace brew::trajectory_usggiw {

// @mex rfs
// @mex_name PHD
// @mex_params prune_threshold:double:1e-4, merge_threshold:double:4.0, max_components:int:100, extract_threshold:double:0.5, gate_threshold:double:9.0
// @mex_init set_intensity
template <typename Scalar = double, int De = Eigen::Dynamic,
          int MaxComponents = Eigen::Dynamic>
class PHD : public multi_target::RFSBase {
    using State = models::TrajectoryUSGGIW<Scalar, De>;
    using Report = models::USGGIW<Scalar, De>;
    using Mixture = models::Mixture<State, MaxComponents>;
    using ReportMixture = models::Mixture<Report>;
public:
    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto out = std::make_unique<PHD>();
        out->prob_detection_ = prob_detection_;
        out->prob_survive_ = prob_survive_;
        out->clutter_rate_ = clutter_rate_;
        out->clutter_density_ = clutter_density_;
        if (filter_) out->filter_ = filter_->clone();
        if (intensity_) out->intensity_ = intensity_->clone();
        if (birth_model_) out->birth_model_ = birth_model_->clone();
        if (reports_) out->reports_ = reports_->clone();
        out->detection_field_ = detection_field_;
        out->detection_groups_ = detection_groups_;
        out->detection_radius_ = detection_radius_;
        out->prob_detection_out_ = prob_detection_out_;
        out->prune_threshold_ = prune_threshold_;
        out->merge_threshold_ = merge_threshold_;
        out->max_components_ = max_components_;
        out->extract_threshold_ = extract_threshold_;
        out->gate_threshold_ = gate_threshold_;
        return out;
    }

    void set_filter(std::unique_ptr<filters::Filter<State>> filter) {
        filter_ = std::move(filter);
    }
    void set_intensity(std::unique_ptr<Mixture> intensity) {
        intensity_ = std::move(intensity);
    }
    void set_birth_model(std::unique_ptr<Mixture> birth) {
        birth_model_ = std::move(birth);
    }
    void set_measurement_components(const ReportMixture& reports) {
        reports_ = reports.clone();
    }
    void set_detection_field(const Eigen::MatrixXd& field, double radius, double pd_out) {
        detection_field_ = field;
        detection_groups_.clear();
        detection_radius_ = radius;
        prob_detection_out_ = pd_out;
    }
    void set_grouped_detection_field(const Eigen::MatrixXd& field,
                                     const std::vector<int>& groups,
                                     double radius, double pd_out) {
        detection_field_ = field;
        detection_groups_ = groups;
        if (detection_groups_.size() != static_cast<std::size_t>(field.cols()))
            detection_groups_.clear();
        detection_radius_ = radius;
        prob_detection_out_ = pd_out;
    }
    void set_prune_threshold(double value) { prune_threshold_ = value; }
    void set_merge_threshold(double value) { merge_threshold_ = value; }
    void set_max_components(int value) { max_components_ = value; }
    void set_extract_threshold(double value) { extract_threshold_ = value; }
    void set_gate_threshold(double value) { gate_threshold_ = value; }

    [[nodiscard]] const Mixture& intensity() const { return *intensity_; }
    [[nodiscard]] const std::deque<std::unique_ptr<Mixture>>& extracted_mixtures() const {
        return extracted_;
    }

    void predict(int, double dt) override {
        if (!intensity_ || !filter_) return;
        for (std::size_t i = 0; i < intensity_->size(); ++i)
            intensity_->weights()(static_cast<Eigen::Index>(i)) *= prob_survive_;
        filter_->predict_batch_dynamic(dt, *intensity_);
        if (birth_model_) intensity_->add_components(*birth_model_->clone());
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !filter_) return;
        auto* trajectory_filter =
            dynamic_cast<filters::TrajectoryUSGGIWEKF<Scalar, De>*>(filter_.get());
        const bool component_mode = trajectory_filter && reports_
            && reports_->size() == static_cast<std::size_t>(measurements.cols());

        Mixture missed;
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            const double pd = pd_at(intensity_->component(i));
            const double prior_weight = intensity_->weight(i);
            if (prior_weight * (1.0 - pd) > 0.0)
                missed.add_component(intensity_->component(i).clone_typed(),
                                     prior_weight * (1.0 - pd));
            const auto& current = intensity_->component(i).current();
            const double zero_count_probability = std::pow(
                current.beta() / (current.beta() + 1.0), current.alpha());
            if (prior_weight * pd * zero_count_probability > 0.0) {
                auto zero_count = intensity_->component(i).clone_typed();
                zero_count->current().beta() += 1.0;
                missed.add_component(std::move(zero_count),
                                     prior_weight * pd * zero_count_probability);
            }
            intensity_->weights()(static_cast<Eigen::Index>(i)) *= pd;
        }

        Mixture detected;
        for (int measurement_index = 0; measurement_index < measurements.cols(); ++measurement_index) {
            const Eigen::VectorXd measurement = measurements.col(measurement_index);
            std::vector<double> weights;
            std::vector<std::unique_ptr<State>> states;
            for (std::size_t state_index = 0; state_index < intensity_->size(); ++state_index) {
                const auto& predicted = intensity_->component(state_index);
                const double distance = component_mode
                    ? trajectory_filter->gate_component(
                        reports_->component(static_cast<std::size_t>(measurement_index)), predicted)
                    : filter_->gate(measurement, predicted);
                if (distance >= gate_threshold_) continue;
                auto result = component_mode
                    ? trajectory_filter->correct_component(
                        reports_->component(static_cast<std::size_t>(measurement_index)), predicted)
                    : filter_->correct(measurement, predicted);
                weights.push_back(result.likelihood * intensity_->weight(state_index));
                states.push_back(result.distribution.clone_typed());
            }
            double sum = 0.0;
            for (double weight : weights) sum += weight;
            const double denominator = std::max(
                clutter_rate_ * clutter_density_ + sum, 1e-300);
            for (std::size_t i = 0; i < weights.size(); ++i)
                detected.add_component(std::move(states[i]), weights[i] / denominator);
        }

        *intensity_ = std::move(missed);
        intensity_->add_components(detected);
        reports_.reset();
    }

    void cleanup() override {
        if (!intensity_) return;
        fusion::prune(*intensity_, prune_threshold_);
        merge(*intensity_, merge_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));
        this->push_history(extracted_, extract());
    }

    [[nodiscard]] std::unique_ptr<Mixture> extract() const {
        auto result = std::make_unique<Mixture>();
        if (!intensity_) return result;
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            if (intensity_->weight(i) >= extract_threshold_)
                result->add_component(intensity_->component(i).clone_typed(), intensity_->weight(i));
        }
        return result;
    }

private:
    [[nodiscard]] double pd_at(const State& trajectory) const {
        if (detection_radius_ <= 0.0 || detection_field_.cols() == 0)
            return prob_detection_;
        const Eigen::Vector3d direction =
            trajectory.current().direction().template cast<double>();
        std::vector<int> visible_groups;
        const double cosine_radius = std::cos(detection_radius_);
        for (int i = 0; i < detection_field_.cols(); ++i) {
            const Eigen::Vector3d center = usggiw::normalized(detection_field_.col(i));
            if (direction.dot(center) < cosine_radius) continue;
            const int group = detection_groups_.empty()
                ? i : detection_groups_[static_cast<std::size_t>(i)];
            if (std::find(visible_groups.begin(), visible_groups.end(), group)
                    == visible_groups.end())
                visible_groups.push_back(group);
        }
        double missed = 1.0;
        for (std::size_t i = 0; i < visible_groups.size(); ++i)
            missed *= 1.0 - prob_detection_;
        return std::max(prob_detection_out_, 1.0 - missed);
    }

    std::unique_ptr<filters::Filter<State>> filter_;
    std::unique_ptr<Mixture> intensity_;
    std::unique_ptr<Mixture> birth_model_;
    std::unique_ptr<ReportMixture> reports_;
    Eigen::MatrixXd detection_field_;
    std::vector<int> detection_groups_;
    double detection_radius_ = -1.0;
    double prob_detection_out_ = 0.0;
    double prune_threshold_ = 1e-4;
    double merge_threshold_ = 4.0;
    int max_components_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    std::deque<std::unique_ptr<Mixture>> extracted_;
};

}  // namespace brew::trajectory_usggiw
