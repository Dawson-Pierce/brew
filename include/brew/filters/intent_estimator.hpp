#pragma once
#include "brew/models/categorical_intent.hpp"

namespace brew::filters {
template<class Dist, class State = models::CategoricalIntent>
class IntentEstimator {
public:
    using Density = Dist;
    using StateType = State;
    double dt = 30.0, objective_stay = 0.9;
    std::vector<models::IntentHypothesis<Dist>> hypotheses;
    IntentEstimator() = default;
    explicit IntentEstimator(std::vector<models::IntentHypothesis<Dist>> hypotheses,
                                        double dt = 30.0, double objective_stay = 0.9)
        : dt(dt), objective_stay(objective_stay), hypotheses(std::move(hypotheses)) { validate(); }
    virtual ~IntentEstimator() = default;

    std::vector<std::string> objectives() const {
        std::vector<std::string> names;
        for (const auto& h : hypotheses) names.push_back(h.name);
        return names;
    }
    virtual void validate() const {
        if (hypotheses.empty() || !std::isfinite(dt) || dt <= 0 || !std::isfinite(objective_stay)
                || objective_stay < 0 || objective_stay > 1)
            throw std::invalid_argument("Invalid categorical intent settings");
        for (std::size_t i = 0; i < hypotheses.size(); ++i) {
            hypotheses[i].validate();
            for (std::size_t j = 0; j < i; ++j)
                if (hypotheses[i].name == hypotheses[j].name) throw std::invalid_argument("Duplicate intent hypothesis");
        }
    }
    static const State& state(const Dist& density) { return density.template augmentation<State>(); }
    static void set_state(Dist& density, State value) { density.set_augmentation(std::move(value)); }

    Eigen::VectorXd predict_probabilities(const Eigen::VectorXd& posterior) const {
        if (posterior.size() != static_cast<int>(hypotheses.size())) throw std::invalid_argument("Intent vector size mismatch");
        if (posterior.size() == 1) return posterior;
        Eigen::MatrixXd transition = Eigen::MatrixXd::Constant(posterior.size(), posterior.size(),
            (1 - objective_stay) / (posterior.size() - 1));
        transition.diagonal().setConstant(objective_stay);
        return transition.transpose() * posterior;
    }
    static Eigen::VectorXd normalize(const Eigen::VectorXd& logs) {
        const double largest = logs.maxCoeff();
        if (!std::isfinite(largest)) throw std::invalid_argument("Intent evidence has no finite support");
        const double normalizer = largest + std::log((logs.array() - largest).exp().sum());
        return (logs.array() - normalizer).exp().matrix();
    }
    Eigen::VectorXd aligned_probabilities(const State& mark) const {
        if (mark.probabilities.size() == 0) return Eigen::VectorXd::Constant(hypotheses.size(), 1.0 / hypotheses.size());
        mark.validate();
        if (mark.probabilities.size() != static_cast<int>(mark.objectives.size()))
            throw std::invalid_argument("Intent probabilities do not match their labels");
        if (mark.objectives == objectives()) return mark.probabilities;
        if (mark.objectives.size() != hypotheses.size()) throw std::invalid_argument("Intent hypotheses changed during recursion");
        Eigen::VectorXd result(hypotheses.size());
        for (std::size_t i = 0; i < hypotheses.size(); ++i) {
            const auto found = std::find(mark.objectives.begin(), mark.objectives.end(), hypotheses[i].name);
            if (found == mark.objectives.end()) throw std::invalid_argument("Intent hypotheses changed during recursion");
            result[i] = mark.probabilities[found - mark.objectives.begin()];
        }
        return result;
    }
    void prepare_prediction(Dist& source) const {
        auto mark = state(source);
        mark.previous_observation.reset();
        if (mark.observed) {
            mark.previous_observation = std::make_shared<const models::IntentDensityObservation<Dist>>(source);
        }
        set_state(source, std::move(mark));
    }
    void predict(Dist& predicted, double interval) const {
        validate();
        const auto old = state(predicted);
        auto mark = prior_state();
        mark.objectives = objectives();
        mark.probabilities = aligned_probabilities(old);
        if (interval > 0.0) mark.probabilities = predict_probabilities(mark.probabilities);
        mark.predicted_probabilities = mark.probabilities;
        mark.previous_observation = old.previous_observation;
        mark.interval_dt = interval > 0.0 ? interval : dt;
        set_state(predicted, std::move(mark));
    }
    void missed_detection(Dist& missed) const {
        auto mark = state(missed);
        if (mark.probabilities.size() == 0) {
            mark = prior_state(); mark.objectives = objectives();
            mark.probabilities = aligned_probabilities(mark);
            mark.predicted_probabilities = mark.probabilities;
        }
        mark.observed = false;
        mark.previous_observation.reset();
        set_state(missed, std::move(mark));
    }
    void correct_branch(Dist& current, const Dist& predicted) const {
        validate();
        const auto seed = state(predicted);
        const Eigen::VectorXd prior = seed.predicted_probabilities.size()
            ? seed.predicted_probabilities : aligned_probabilities(seed);
        if (prior.size() != static_cast<int>(hypotheses.size())) throw std::invalid_argument("Intent prior size mismatch");
        const auto* observation = dynamic_cast<const models::IntentDensityObservation<Dist>*>(seed.previous_observation.get());
        const Dist* previous = observation ? &observation->density : nullptr;
        auto mark = observation_state(current);
        mark.objectives = objectives();
        mark.probabilities = posterior(current, previous, seed.interval_dt > 0.0 ? seed.interval_dt : dt, prior);
        mark.predicted_probabilities = prior;
        mark.previous_observation = seed.previous_observation;
        mark.interval_dt = seed.interval_dt;
        mark.observed = true;
        set_state(current, std::move(mark));
    }
    // Single-object recursion; a new density starts with a uniform intent prior.
    void correct(Dist& current, const Dist* previous,
                 bool continuous_observation = true) const {
        validate();
        if (!previous || state(*previous).probabilities.size() == 0) {
            State mark;
            mark.objectives = objectives();
            mark.probabilities = Eigen::VectorXd::Constant(hypotheses.size(), 1.0 / hypotheses.size());
            set_state(current, std::move(mark));
            return;
        }
        if (state(*previous).objectives != objectives()
            || state(*previous).probabilities.size() != static_cast<int>(hypotheses.size()))
            throw std::invalid_argument("Intent hypotheses changed during recursion");
        State mark;
        mark.objectives = objectives();
        mark.probabilities = posterior(current, continuous_observation ? previous : nullptr, dt,
                                      predict_probabilities(state(*previous).probabilities));
        set_state(current, std::move(mark));
    }

    // A known single-object history needs no inter-object association. This uses
    // the same recursion for journal evidence ablations and offline analysis.
    Eigen::MatrixXd posterior_history(const std::vector<Dist>& observations) const {
        validate();
        Eigen::MatrixXd result(observations.size(), hypotheses.size());
        std::optional<Dist> previous;
        for (std::size_t i = 0; i < observations.size(); ++i) {
            auto current = observations[i];
            correct(current, previous ? &*previous : nullptr);
            result.row(i) = state(current).probabilities.transpose();
            previous = std::move(current);
        }
        return result;
    }

    Eigen::MatrixXd evidence_recursion(const Eigen::MatrixXd& log_evidence) const {
        validate();
        BREW_ASSERT(log_evidence.cols() == static_cast<int>(hypotheses.size()),
                    "Evidence columns do not match objective list");
        Eigen::MatrixXd result(log_evidence.rows(), log_evidence.cols());
        if (!result.rows()) return result;
        result.row(0).setConstant(1.0 / hypotheses.size());
        for (int i = 1; i < result.rows(); ++i) {
            Eigen::VectorXd previous = result.row(i - 1).transpose();
            Eigen::VectorXd logs = predict_probabilities(previous).array().log();
            logs += log_evidence.row(i).transpose();
            result.row(i) = normalize(logs).transpose();
        }
        return result;
    }
private:
    Eigen::VectorXd posterior(const Dist& current, const Dist* previous, double interval,
                              const Eigen::VectorXd& prior) const {
        Eigen::VectorXd logs = prior.array().log();
        for (int i = 0; i < logs.size(); ++i)
            for (double term : hypotheses[i].model->log_evidence(current, previous, interval)) logs[i] += term;
        return normalize(logs);
    }
protected:
    virtual State prior_state() const { return {}; }
    virtual State observation_state(const Dist&) const { return prior_state(); }
};

}
