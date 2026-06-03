#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <variant>

#include "brew/dynamics/dynamics_base.hpp"
#include "brew/shared/mixture.hpp"
#include <cstddef>

namespace brew::filters {

template <typename Dist>
class Filter {
public:
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

    [[nodiscard]] virtual Dist predict(double dt, const Dist& prev) const = 0;

    [[nodiscard]] virtual CorrectionResult correct(
        const MeasVector& measurement,
        const Dist& predicted) const = 0;

    [[nodiscard]] virtual double gate(
        const MeasVector& measurement,
        const Dist& predicted) const = 0;

    template <int N>
    void predict_batch(double dt, models::Mixture<Dist, N>& mix) const {
        const std::size_t K = mix.size();
        for (std::size_t k = 0; k < K; ++k) {
            mix.component(k) = this->predict(dt, mix.component(k));
        }
    }

    virtual void predict_batch_dynamic(
        double dt, models::Mixture<Dist, Eigen::Dynamic>& mix) const {
        this->predict_batch(dt, mix);
    }

protected:
    std::shared_ptr<DynamicsType> dyn_obj_;
    MeasurementFunc h_;
    MeasurementJacobian H_;
    StateMatrix process_noise_;
    MeasNoiseMatrix measurement_noise_;
};

}
