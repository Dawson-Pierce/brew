#pragma once

#include "brew/utils/clustering/dbscan.hpp"
#include "brew/filters/filter_base.hpp"
#include "brew/utils/fusion/cap.hpp"
#include "brew/utils/fusion/prune.hpp"
#include "brew/multiObjectTrackers/rfs_base.hpp"
#include <algorithm>
#include <deque>
#include <map>
#include <optional>

namespace brew::multiObjectTrackers {

// The density and its single-object filter supply all state-space mathematics.
// @mex rfs
// @mex_name PHD
// @mex_params prune_threshold:double:1e-4, merge_threshold:double:4.0, merge_rotation_threshold:double:1.5707963267948966, max_components:int:100, extract_threshold:double:0.5, gate_threshold:double:9.0
// @mex_init set_intensity
// @mex_has birth_weights, cluster_object
// @mex_config prob_detection, prob_survive, clutter_rate, clutter_density, max_history, prune_threshold, merge_threshold, merge_rotation_threshold, max_components, extract_threshold, gate_threshold, cluster_object, birth_model, intensity
template<class T, int MaxComponents = Eigen::Dynamic>
class PHD : public RFSBase {
public:
    using Density = T;
    using Filter = filters::Filter<T>;
    using Report = typename Filter::Report;
    using Mixture = models::Mixture<T, MaxComponents>;
    using ReportMixture = models::Mixture<Report>;

    struct Config {
        std::optional<double> prob_detection, prob_survive, clutter_rate, clutter_density;
        std::optional<std::size_t> max_history;
        std::optional<double> prune_threshold, merge_threshold, merge_rotation_threshold;
        std::optional<int> max_components;
        std::optional<double> extract_threshold, gate_threshold;
        std::shared_ptr<utils::clustering::ClusterBase> cluster_object;
        // Borrowed only during construction; the PHD clones both mixtures.
        const Mixture* birth_model = nullptr;
        const Mixture* intensity = nullptr;
    };

    PHD() = default;
    explicit PHD(const Filter& filter) : filter_(filter.clone()),
        intensity_(std::make_unique<Mixture>()), birth_model_(std::make_unique<Mixture>()) {}
    PHD(const Filter& filter, const Config& c) : PHD(filter) {
        if (c.prob_detection) set_prob_detection(*c.prob_detection);
        if (c.prob_survive) set_prob_survive(*c.prob_survive);
        if (c.clutter_rate) set_clutter_rate(*c.clutter_rate);
        if (c.clutter_density) set_clutter_density(*c.clutter_density);
        if (c.max_history) set_max_history(*c.max_history);
        if (c.prune_threshold) set_prune_threshold(*c.prune_threshold);
        if (c.merge_threshold) set_merge_threshold(*c.merge_threshold);
        if (c.merge_rotation_threshold) set_merge_rotation_threshold(*c.merge_rotation_threshold);
        if (c.max_components) set_max_components(*c.max_components);
        if (c.extract_threshold) set_extract_threshold(*c.extract_threshold);
        if (c.gate_threshold) set_gate_threshold(*c.gate_threshold);
        if (c.birth_model) set_birth_model(c.birth_model->clone());
        if (c.intensity) set_intensity(c.intensity->clone());
        if (c.cluster_object) set_cluster_object(c.cluster_object);
    }

    std::unique_ptr<RFSBase> clone() const override {
        auto result = std::make_unique<PHD>();
        result->prob_detection_ = prob_detection_;
        result->prob_survive_ = prob_survive_;
        result->clutter_rate_ = clutter_rate_;
        result->clutter_density_ = clutter_density_;
        result->max_history_ = max_history_;
        if (filter_) result->filter_ = filter_->clone();
        if (intensity_) result->intensity_ = intensity_->clone();
        if (birth_model_) result->birth_model_ = birth_model_->clone();
        if (reports_) result->reports_ = reports_->clone();
        result->cluster_ = cluster_;
        result->extended_ = extended_;
        result->prune_threshold_ = prune_threshold_;
        result->merge_threshold_ = merge_threshold_;
        result->rotation_threshold_ = rotation_threshold_;
        result->extract_threshold_ = extract_threshold_;
        result->gate_threshold_ = gate_threshold_;
        result->max_components_ = max_components_;
        result->field_ = field_;
        result->groups_ = groups_;
        result->radius_ = radius_;
        result->pd_out_ = pd_out_;
        for (const auto& snapshot : extracted_) result->extracted_.push_back(snapshot->clone());
        return result;
    }

    void set_filter(std::unique_ptr<Filter> filter) { filter_ = std::move(filter); }
    Filter& filter() { return *filter_; }
    const Filter& filter() const { return *filter_; }
    void set_intensity(std::unique_ptr<Mixture> mixture) { intensity_ = std::move(mixture); }
    void set_birth_model(std::unique_ptr<Mixture> births) {
        if (births && !births->empty()) {
            extended_ = births->component(0).is_extended();
            if (extended_ && !cluster_ && (!filter_ || filter_->groups_observations())) cluster_ = std::make_shared<utils::clustering::DBSCAN>();
        }
        birth_model_ = std::move(births);
    }
    void set_birth_weights(const Eigen::VectorXd& weights) {
        if (!birth_model_) return;
        if (weights.size() == 1) birth_model_->weights().setConstant(weights(0));
        else birth_model_->weights() = weights;
    }
    void set_measurement_components(const ReportMixture& reports) { reports_ = reports.clone(); }
    void set_prune_threshold(double value) { prune_threshold_ = value; }
    void set_merge_threshold(double value) { merge_threshold_ = value; }
    void set_merge_rotation_threshold(double value) { rotation_threshold_ = value; }
    void set_extract_threshold(double value) { extract_threshold_ = value; }
    void set_gate_threshold(double value) { gate_threshold_ = value; }
    void set_max_components(int value) { max_components_ = value; }
    void set_extended_target(bool value) { extended_ = value; }
    void set_cluster_object(std::shared_ptr<utils::clustering::ClusterBase> cluster) {
        cluster_ = std::move(cluster);
        extended_ = true;
    }
    void set_detection_field(const Eigen::MatrixXd& field, double radius, double pd_out) {
        set_grouped_detection_field(field, {}, radius, pd_out);
    }
    void set_grouped_detection_field(const Eigen::MatrixXd& field, const std::vector<int>& groups,
            double radius, double pd_out) {
        field_ = field;
        groups_ = groups.size() == static_cast<std::size_t>(field.cols()) ? groups : std::vector<int>{};
        radius_ = radius;
        pd_out_ = pd_out;
    }
    double detection_radius() const { return radius_; }
    double outside_detection_probability() const { return pd_out_; }
    double extraction_threshold() const { return extract_threshold_; }
    const Mixture& intensity() const { return *intensity_; }
    Mixture& intensity() { return *intensity_; }
    const auto& extracted_mixtures() const { return extracted_; }

    void predict(int, double dt) override {
        if (!intensity_ || !filter_) return;
        intensity_->weights() *= prob_survive_;
        filter_->predict_mixture(dt, *intensity_);
        if (birth_model_) intensity_->add_components(*birth_model_);
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !filter_) return;
        const bool component_mode = reports_ && reports_->size() == static_cast<std::size_t>(measurements.cols());
        std::vector<Eigen::MatrixXd> measurements_by_object;
        if (!component_mode && extended_ && cluster_ && filter_->groups_observations()) measurements_by_object = cluster_->cluster(measurements);
        else for (Eigen::Index i = 0; i < measurements.cols(); ++i)
            measurements_by_object.push_back(measurements.col(i));

        Mixture missed;
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            const auto& prior = intensity_->component(i);
            const double pd = filter_->detection_probability(prior, field_, radius_,
                prob_detection_, pd_out_, groups_);
            for (auto update : filter_->missed_detection(prior, pd, intensity_->weight(i))) {
                const double weight = update.weight;
                missed.add_component(std::move(update.distribution), weight);
            }
            intensity_->weights()(i) *= pd;
        }

        Mixture detected;
        for (std::size_t z = 0; z < measurements_by_object.size(); ++z) {
            const auto& group = measurements_by_object[z];
            const auto center = filter_->gate_center(group);
            const Eigen::VectorXd packed = Eigen::Map<const Eigen::VectorXd>(group.data(), group.size());
            std::vector<bool> accepted(intensity_->size(), false);
            std::vector<const T*> priors;
            for (std::size_t i = 0; i < intensity_->size(); ++i) {
                const auto& prior = intensity_->component(i);
                priors.push_back(&prior);
                const double distance = component_mode
                    ? filter_->gate_report(reports_->component(z), prior) : filter_->gate(center, prior);
                accepted[i] = distance < gate_threshold_;
            }
            filter_->expand_gates(accepted, priors);
            std::vector<const T*> gated;
            for (std::size_t i = 0; i < priors.size(); ++i)
                if (accepted[i]) gated.push_back(priors[i]);
            std::vector<typename Filter::LogCorrectionResult> corrections;
            if (!component_mode) corrections = filter_->correct_batch_with_log(packed, gated);
            std::size_t correction_index = 0;
            std::vector<std::unique_ptr<T>> states;
            std::vector<double> weights, prior_weights, linear_weights;
            bool logarithmic = filter_->log_domain_correction();
            for (std::size_t i = 0; i < intensity_->size(); ++i) {
                if (!accepted[i]) continue;
                const double prior_weight = intensity_->weight(i);
                if (!component_mode) {
                    auto [update, log_likelihood] = std::move(corrections[correction_index++]);
                    states.push_back(update.distribution.clone_typed());
                    const double linear = prior_weight * update.likelihood;
                    linear_weights.push_back(linear);
                    logarithmic = logarithmic || !std::isfinite(linear)
                        || (prior_weight > 0. && std::isfinite(log_likelihood) && (linear == 0.
                            || std::log(update.likelihood) != log_likelihood));
                    weights.push_back(prior_weight > 0. ? std::log(prior_weight) + log_likelihood
                        : -std::numeric_limits<double>::infinity());
                } else {
                    auto update = component_mode ? filter_->correct_report(reports_->component(z), *priors[i])
                                                 : filter_->correct(packed, *priors[i]);
                    states.push_back(update.distribution.clone_typed());
                    const double weight = prior_weight * update.likelihood;
                    linear_weights.push_back(weight);
                    logarithmic = logarithmic || !std::isfinite(weight);
                    weights.push_back(weight > 0. ? std::log(weight) : -std::numeric_limits<double>::infinity());
                }
                prior_weights.push_back(prior_weight);
            }
            const int count = component_mode ? 1 : static_cast<int>(group.cols());
            const double clutter = clutter_rate_ * clutter_density_;
            if (logarithmic) {
                filter_->adjust_log_weights(states, prior_weights, weights);
                double denominator = clutter > 0. ? count * std::log(clutter)
                    : -std::numeric_limits<double>::infinity();
                for (double weight : weights) {
                    if (!std::isfinite(weight)) continue;
                    if (!std::isfinite(denominator)) denominator = weight;
                    else denominator = std::max(denominator, weight)
                        + std::log1p(std::exp(-std::abs(denominator - weight)));
                }
                if (!std::isfinite(denominator)) continue;
                for (std::size_t i = 0; i < states.size(); ++i)
                    if (std::isfinite(weights[i])) detected.add_component(std::move(states[i]), std::exp(weights[i]-denominator));
            } else {
                double sum = 0.;
                for (double weight : linear_weights) sum += weight;
                const double denominator = std::max((count > 1 ? std::pow(clutter, count) : clutter) + sum, 1e-300);
                for (std::size_t i=0; i<states.size(); ++i)
                    detected.add_component(std::move(states[i]), linear_weights[i]/denominator);
            }

        }
        *intensity_ = std::move(missed);
        intensity_->add_components(detected);
        reports_.reset();
    }

    void cleanup() override {
        if (!intensity_) return;
        intensity_->reduce(prune_threshold_, merge_threshold_, max_components_, rotation_threshold_);
        if (filter_) filter_->finalize_mixture(*intensity_);
        push_history(extracted_, extract());
    }

    std::vector<Report> step(Eigen::MatrixXd measurements, double dt, int frame = 0) {
        if (!filter_) return {};
        models::Mixture<T> previous;
        if (!extracted_.empty()) {
            const auto& snapshot = *extracted_.back();
            for (std::size_t i = 0; i < snapshot.size(); ++i)
                previous.add_component(snapshot.component(i), snapshot.weight(i));
        }
        auto births = filter_->prepare_step(measurements, previous, dt, prune_threshold_, gate_threshold_, reports_);
        if (births) {
            auto converted = std::make_unique<Mixture>();
            for (std::size_t i = 0; i < births->size(); ++i)
                converted->add_component(births->component(i), births->weight(i));
            set_birth_model(std::move(converted));
        }
        predict(frame, dt);
        correct(measurements);
        cleanup();
        return estimates();
    }

    std::unique_ptr<Mixture> extract() const {
        auto result = std::make_unique<Mixture>();
        if (intensity_) for (std::size_t i = 0; i < intensity_->size(); ++i)
            if (intensity_->weight(i) >= extract_threshold_)
                result->add_component(intensity_->component(i), intensity_->weight(i));
        return result;
    }
    std::unique_ptr<Mixture> mixture(std::optional<double> minimum = std::nullopt) const {
        auto result = std::make_unique<Mixture>();
        if (!extracted_.empty()) for (std::size_t i = 0; i < extracted_.back()->size(); ++i)
            if (extracted_.back()->weight(i) >= minimum.value_or(extract_threshold_))
                result->add_component(extracted_.back()->component(i), extracted_.back()->weight(i));
        return result;
    }
    std::vector<Report> estimates() const {
        std::vector<Report> result;
        if (extracted_.empty()) return result;
        const auto selected = selected_indices();
        for (auto i : selected) result.push_back(models::current_density(extracted_.back()->component(i)));
        return result;
    }
    std::vector<Eigen::MatrixXd> current_extracted_histories() const {
        std::vector<Eigen::MatrixXd> result;
        for (auto i : selected_indices()) result.push_back(filter_->history(extracted_.back()->component(i)));
        return result;
    }
    std::map<int,Eigen::MatrixXd> track_histories() const {
        std::map<int,Eigen::MatrixXd> result;
        auto histories = current_extracted_histories();
        for (std::size_t i = 0; i < histories.size(); ++i) result.emplace(i, std::move(histories[i]));
        return result;
    }
    double expected_cardinality() const { return intensity_ ? intensity_->weights().sum() : 0.; }
    double extracted_cardinality() const { return extracted_.empty() ? 0. : extracted_.back()->weights().sum(); }

private:
    std::vector<std::size_t> selected_indices() const {
        std::vector<std::size_t> indices;
        std::vector<const T*> components;
        std::vector<double> weights;
        if (extracted_.empty()) return indices;
        const auto& latest = *extracted_.back();
        for (std::size_t i = 0; i < latest.size(); ++i) {
            components.push_back(&latest.component(i));
            weights.push_back(latest.weight(i));
            if (latest.weight(i) >= extract_threshold_) indices.push_back(i);
        }
        if (filter_) filter_->select_estimates(indices, components, weights);
        return indices;
    }
    std::unique_ptr<Filter> filter_;
    std::unique_ptr<Mixture> intensity_, birth_model_;
    std::unique_ptr<ReportMixture> reports_;
    std::shared_ptr<utils::clustering::ClusterBase> cluster_;
    bool extended_ = false;
    double prune_threshold_ = 1e-4, merge_threshold_ = 4., rotation_threshold_ = 1.5707963267948966;
    double extract_threshold_ = .5, gate_threshold_ = 9.;
    int max_components_ = 100;
    Eigen::MatrixXd field_;
    std::vector<int> groups_;
    double radius_ = -1., pd_out_ = 0.;
    std::deque<std::unique_ptr<Mixture>> extracted_;
};
}


