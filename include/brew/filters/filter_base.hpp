#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <variant>

#include "brew/dynamics/dynamics_base.hpp"
#include "brew/models/mixture.hpp"
#include "brew/filters/filter_traits.hpp"
#include <stdexcept>
#include <vector>
#include <cstddef>

namespace brew::filters {

template <typename Dist>
class Filter {
public:
    using Density = Dist;
    using Report = models::current_density_t<Dist>;
    using DistScalar = typename Dist::Vector::Scalar;
    static constexpr int DistStateDim = Dist::Vector::RowsAtCompileTime;
    using DynamicsType = dynamics::DynamicsBase<DistScalar, DistStateDim>;

    using StateVector      = Eigen::Matrix<DistScalar, DistStateDim, 1>;
    using StateMatrix      = Eigen::Matrix<DistScalar, DistStateDim, DistStateDim>;
    using MeasVector       = Eigen::Matrix<DistScalar, Eigen::Dynamic, 1>;
    using MeasMatrix       = Eigen::Matrix<DistScalar, Eigen::Dynamic, DistStateDim>;
    using MeasNoiseMatrix  = Eigen::Matrix<DistScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using KalmanGainMatrix = Eigen::Matrix<DistScalar, DistStateDim, Eigen::Dynamic>;

    using MeasurementFunc = std::function<MeasVector(const StateVector&)>;
    using MeasurementJacobian = std::variant<MeasMatrix,
        std::function<MeasMatrix(const StateVector&)>>;

    virtual ~Filter() = default;

    [[nodiscard]] virtual std::unique_ptr<Filter<Dist>> clone() const = 0;

    void set_dynamics(std::shared_ptr<DynamicsType> dyn) {
        dyn_obj_ = std::move(dyn);
    }

    void set_measurement_function(MeasurementFunc h) { h_ = std::move(h); }
    void set_measurement_jacobian(MeasurementJacobian H) { H_ = std::move(H); }

    void set_process_noise(StateMatrix Q) { process_noise_ = std::move(Q); }
    void set_measurement_noise(MeasNoiseMatrix R) { measurement_noise_ = std::move(R); }

    [[nodiscard]] const StateMatrix& process_noise() const { return process_noise_; }
    [[nodiscard]] const MeasNoiseMatrix& measurement_noise() const { return measurement_noise_; }
    [[nodiscard]] DynamicsType& dynamics() const { return *dyn_obj_; }

    [[nodiscard]] MeasVector estimate_measurement(const StateVector& state) const {
        if (h_) {
            return h_(state);
        }
        return get_measurement_matrix(state) * state;
    }

    [[nodiscard]] MeasMatrix get_measurement_matrix(const StateVector& state) const {
        if (auto* mat = std::get_if<MeasMatrix>(&H_)) {
            return *mat;
        }
        return std::get<std::function<MeasMatrix(const StateVector&)>>(H_)(state);
    }

    struct CorrectionResult {
        Dist distribution;
        double likelihood;
    };

    using LogCorrectionResult = std::pair<CorrectionResult, double>;

    [[nodiscard]] virtual Dist predict(double dt, const Dist& prev) const = 0;

    [[nodiscard]] virtual CorrectionResult correct(
        const MeasVector& measurement,
        const Dist& predicted) const = 0;

    [[nodiscard]] virtual LogCorrectionResult correct_with_log(
        const MeasVector& measurement,
        const Dist& predicted) const {
        auto result = correct(measurement, predicted);
        const double log_likelihood = result.likelihood > 0.0
            ? std::log(result.likelihood)
            : -std::numeric_limits<double>::infinity();
        return {std::move(result), log_likelihood};
    }

    // Joint preparation may share measurement representations across hypotheses.
    virtual std::vector<LogCorrectionResult> correct_batch_with_log(
            const MeasVector& measurement, const std::vector<const Dist*>& priors) const {
        std::vector<LogCorrectionResult> result;
        result.reserve(priors.size());
        for (const auto* prior : priors) result.push_back(correct_with_log(measurement, *prior));
        return result;
    }

    [[nodiscard]] virtual double gate(
        const MeasVector& measurement,
        const Dist& predicted) const = 0;

    virtual MeasVector gate_center(const Eigen::MatrixXd& group) const {
        return group.rowwise().mean().template cast<DistScalar>();
    }
    virtual bool groups_observations() const { return true; }
    virtual void expand_gates(std::vector<bool>&, const std::vector<const Dist*>&) const {}
    virtual void adjust_log_weights(std::vector<std::unique_ptr<Dist>>&,
            const std::vector<double>&, std::vector<double>&) const {}
    virtual void select_estimates(std::vector<std::size_t>&,
            const std::vector<const Dist*>&, const std::vector<double>&) const {}
    virtual Eigen::MatrixXd history(const Dist& density) const {
        return models::current_density(density).mean().template cast<double>().transpose();
    }
    virtual std::unique_ptr<models::Mixture<Dist>> prepare_step(Eigen::MatrixXd&,
            const models::Mixture<Dist>&, double, double, double,
            std::unique_ptr<models::Mixture<Report>>&) const { return nullptr; }

    virtual void finalize_mixture_dynamic(models::Mixture<Dist>&) {}
    template<int N>
    void finalize_mixture(models::Mixture<Dist,N>& mixture) {
        if constexpr (N == Eigen::Dynamic) finalize_mixture_dynamic(mixture);
        else {
            models::Mixture<Dist> dynamic;
            for (std::size_t i=0; i<mixture.size(); ++i)
                dynamic.add_component(mixture.component(i), mixture.weight(i));
            finalize_mixture_dynamic(dynamic);
            mixture.clear();
            for (std::size_t i=0; i<dynamic.size(); ++i)
                mixture.add_component(dynamic.component(i), dynamic.weight(i));
        }
    }

    struct WeightedDensity { Dist distribution; double weight; };
    virtual bool log_domain_correction() const { return false; }

    // An empty observation is a single-object update, just like a detection.
    // A missed observation can produce a weighted mixture for one prior component.
    virtual std::vector<WeightedDensity> missed_detection(const Dist& prior, double pd, double weight) const {
        return {{prior, weight * (1.0 - pd)}};
    }

    template<int N>
    std::pair<std::unique_ptr<models::Mixture<Dist, N>>, double> missed_mixture(
            const models::Mixture<Dist, N>& source, double pd, bool normalize = true) const {
        auto result = std::make_unique<models::Mixture<Dist, N>>();
        double total = 0.;
        for (std::size_t i = 0; i < source.size(); ++i) {
            for (auto update : missed_detection(source.component(i), pd, source.weight(i))) {
                const double weight = update.weight;
                if (weight > 0.) {
                    result->add_component(std::move(update.distribution), weight);
                    total += weight;
                }
            }
        }
        if (normalize && total > 0.) result->weights() /= total;
        return {std::move(result), total};
    }

    virtual CorrectionResult correct_report(const Report&, const Dist&) const {
        throw std::logic_error("This single-object filter does not accept density reports");
    }
    virtual double gate_report(const Report&, const Dist&) const {
        throw std::logic_error("This single-object filter does not accept density reports");
    }

    virtual double detection_probability(const Dist& density, const Eigen::MatrixXd& field,
            double radius, double pd, double pd_out, const std::vector<int>& = {}) const {
        if (radius <= 0. || field.cols() == 0) return pd;
        const auto& current = models::current_density(density);
        for (Eigen::Index i = 0; i < field.cols(); ++i)
            if ((current.mean().head(field.rows()).template cast<double>() - field.col(i)).norm() <= radius)
                return pd;
        return pd_out;
    }

    // Fixed capacity storage still uses the filter's branching prediction operation.
    template<int N>
    void predict_mixture(double dt, models::Mixture<Dist, N>& mixture) const {
        if constexpr (N == Eigen::Dynamic) predict_batch_dynamic(dt, mixture);
        else {
            models::Mixture<Dist> dynamic;
            for (std::size_t i = 0; i < mixture.size(); ++i)
                dynamic.add_component(mixture.component(i), mixture.weight(i));
            predict_batch_dynamic(dt, dynamic);
            mixture.clear();
            for (std::size_t i = 0; i < dynamic.size(); ++i)
                mixture.add_component(dynamic.component(i), dynamic.weight(i));
        }
    }

    template <int N>
    void predict_batch(double dt, models::Mixture<Dist, N>& mix) const {
        if (predict_impulses(dt, mix)) return;
        const std::size_t K = mix.size();
        for (std::size_t k = 0; k < K; ++k) {
            auto predicted = this->predict(dt, mix.component(k));
            if constexpr (requires { predicted.copy_augmentation_from(mix.component(k)); })
                predicted.copy_augmentation_from(mix.component(k));
            mix.component(k) = std::move(predicted);
        }
    }

    virtual void predict_batch_dynamic(
        double dt, models::Mixture<Dist, Eigen::Dynamic>& mix) const {
        this->predict_batch(dt, mix);
    }

protected:
    // Ordinary Euclidean densities use the same dynamics-owned impulse response.
    // Pose filters override this operation to include attitude cross-covariances.
    virtual Dist predict_impulse_branch(double dt, const Dist& previous,
            const dynamics::ImpulseModel& model, int branch, int) const {
        if constexpr (requires(Dist& d) { d.mean() += StateVector{}; d.covariance() += StateMatrix{}; }
                      && !requires(const Dist& d) { d.current(); }) {
            if (model.body_frame())
                throw std::invalid_argument("Body-frame impulse prediction requires an attitude-aware filter");
            auto result = this->predict(dt, previous);
            if (branch == 1) {
                const auto effect = model.prediction(
                    dyn_obj_->get_state_mat(dt, previous.mean()).template cast<double>());
                result.mean() += effect.mean_shift.template cast<DistScalar>();
                result.covariance() += effect.covariance.template cast<DistScalar>();
            }
            return result;
        } else {
            throw std::invalid_argument("This density needs a filter-specific impulse prediction operation");
        }
    }

    template<int N>
    bool predict_impulses(double dt, models::Mixture<Dist, N>& mixture) const {
        const auto* impulse = dyn_obj_ ? dyn_obj_->impulse_model() : nullptr;
        if (!impulse || dt <= 0.) return false;
        models::Mixture<Dist, N> expanded;
        for (std::size_t i = 0; i < mixture.size(); ++i) {
            for (int branch = 0; branch < 2; ++branch) {
                auto predicted = predict_impulse_branch(dt, mixture.component(i), *impulse, branch,
                                                       static_cast<int>(i));
                if constexpr (requires { predicted.copy_augmentation_from(mixture.component(i)); })
                    predicted.copy_augmentation_from(mixture.component(i));
                expanded.add_component(std::move(predicted),
                    mixture.weight(i) * impulse->branch_probability(branch));
            }
        }
        mixture = std::move(expanded);
        return true;
    }

    std::shared_ptr<DynamicsType> dyn_obj_;
    MeasurementFunc h_;
    MeasurementJacobian H_;
    StateMatrix process_noise_;
    MeasNoiseMatrix measurement_noise_;
};

}
