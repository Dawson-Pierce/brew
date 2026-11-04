#pragma once

#include "brew/filters/filter_base.hpp"

namespace brew::filters {

// An RFS filter composed of an independently configured primary filter and an
// intent estimator. Configure the primary before wrapping; both are copied.
template <typename Dist, typename IntentEstimator>
class IntentFilter final : public Filter<Dist> {
public:
    using Base = Filter<Dist>;
    using typename Base::CorrectionResult;
    using typename Base::LogCorrectionResult;
    using typename Base::MeasVector;
    using typename Base::Report;
    using typename Base::WeightedDensity;

    IntentFilter(const Base& primary, const IntentEstimator& intent)
        : primary_(primary.clone()), intent_(intent) {
        intent_.validate();
    }

    IntentFilter(const IntentFilter& other)
        : primary_(other.primary_->clone()), intent_(other.intent_) {}

    std::unique_ptr<Base> clone() const override {
        return std::make_unique<IntentFilter>(*this);
    }

    Dist predict(double dt, const Dist& previous) const override {
        auto source = previous;
        intent_.prepare_prediction(source);
        auto predicted = primary_->predict(dt, source);
        predicted.copy_augmentation_from(source);
        intent_.predict(predicted, dt);
        return predicted;
    }
    void predict_batch_dynamic(double dt, models::Mixture<Dist>& mixture) const override {
        // Capture each parent before primary prediction can split it. Children
        // inherit that immutable context through the density's ordinary copy.
        for (std::size_t i = 0; i < mixture.size(); ++i)
            intent_.prepare_prediction(mixture.component(i));
        primary_->predict_batch_dynamic(dt, mixture);
        for (std::size_t i = 0; i < mixture.size(); ++i)
            intent_.predict(mixture.component(i), dt);
    }
    CorrectionResult correct(const MeasVector& measurement, const Dist& predicted) const override {
        auto result = primary_->correct(measurement, predicted);
        intent_.correct_branch(result.distribution, predicted);
        return result;
    }
    LogCorrectionResult correct_with_log(const MeasVector& measurement, const Dist& predicted) const override {
        auto result = primary_->correct_with_log(measurement, predicted);
        intent_.correct_branch(result.first.distribution, predicted);
        return result;
    }
    std::vector<LogCorrectionResult> correct_batch_with_log(
            const MeasVector& measurement, const std::vector<const Dist*>& priors) const override {
        auto result = primary_->correct_batch_with_log(measurement, priors);
        for (std::size_t i = 0; i < result.size(); ++i)
            intent_.correct_branch(result[i].first.distribution, *priors[i]);
        return result;
    }
    double gate(const MeasVector& measurement, const Dist& predicted) const override {
        return primary_->gate(measurement, predicted);
    }
    MeasVector gate_center(const Eigen::MatrixXd& group) const override {
        return primary_->gate_center(group);
    }
    bool groups_observations() const override { return primary_->groups_observations(); }
    bool log_domain_correction() const override { return primary_->log_domain_correction(); }
    void expand_gates(std::vector<bool>& accepted, const std::vector<const Dist*>& priors) const override {
        primary_->expand_gates(accepted, priors);
    }
    void adjust_log_weights(std::vector<std::unique_ptr<Dist>>& states,
            const std::vector<double>& prior_weights, std::vector<double>& log_weights) const override {
        primary_->adjust_log_weights(states, prior_weights, log_weights);
        // Primary filters may complete shared branch evidence here (e.g. the
        // no-burn/burn pair). Re-evaluate from the SAME HMM prior, never apply a
        // second transition or multiply the same likelihood twice.
        for (auto& state : states) intent_.correct_branch(*state, *state);
    }
    std::vector<WeightedDensity> missed_detection(const Dist& prior, double pd, double weight) const override {
        auto result = primary_->missed_detection(prior, pd, weight);
        for (auto& branch : result) {
            branch.distribution.copy_augmentation_from(prior);
            intent_.missed_detection(branch.distribution);
        }
        return result;
    }
    CorrectionResult correct_report(const Report& report, const Dist& predicted) const override {
        auto result = primary_->correct_report(report, predicted);
        intent_.correct_branch(result.distribution, predicted);
        return result;
    }
    double gate_report(const Report& report, const Dist& predicted) const override {
        return primary_->gate_report(report, predicted);
    }
    double detection_probability(const Dist& density, const Eigen::MatrixXd& field,
            double radius, double pd, double pd_out, const std::vector<int>& indices = {}) const override {
        return primary_->detection_probability(density, field, radius, pd, pd_out, indices);
    }
    void select_estimates(std::vector<std::size_t>& selected,
            const std::vector<const Dist*>& states, const std::vector<double>& weights) const override {
        primary_->select_estimates(selected, states, weights);
    }
    Eigen::MatrixXd history(const Dist& density) const override { return primary_->history(density); }
    std::unique_ptr<models::Mixture<Dist>> prepare_step(Eigen::MatrixXd& measurements,
            const models::Mixture<Dist>& previous, double dt, double prune, double gate,
            std::unique_ptr<models::Mixture<Report>>& reports) const override {
        return primary_->prepare_step(measurements, previous, dt, prune, gate, reports);
    }

    void finalize_mixture_dynamic(models::Mixture<Dist>& mixture) override {
        primary_->finalize_mixture_dynamic(mixture);
    }

private:
    std::unique_ptr<Base> primary_;
    IntentEstimator intent_;
};

} // namespace brew::filters
