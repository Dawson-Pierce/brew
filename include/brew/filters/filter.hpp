#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <variant>

#include "brew/dynamics/dynamics_base.hpp"

namespace brew::filters {

/// Measurement model types
using MeasurementFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& state)>;
using MeasurementJacobian = std::variant<Eigen::MatrixXd,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& state)>>;

/// Base filter template. Specializations define predict/correct for each T.
/// T is the distribution type (Gaussian, GGIW, TrajectoryGaussian, TrajectoryGGIW).
template <typename T>
class Filter {
public:
    virtual ~Filter() = default;

    [[nodiscard]] virtual std::unique_ptr<Filter<T>> clone() const = 0;

    // ---- Model configuration ----

    void set_dynamics(std::shared_ptr<dynamics::DynamicsBase> dyn) {
        dyn_obj_ = std::move(dyn);
    }

    void set_measurement_function(MeasurementFunc h) { h_ = std::move(h); }
    void set_measurement_jacobian(MeasurementJacobian H) { H_ = std::move(H); }

    void set_process_noise(Eigen::MatrixXd Q) { process_noise_ = std::move(Q); }
    void set_measurement_noise(Eigen::MatrixXd R) { measurement_noise_ = std::move(R); }

    // ---- Accessors ----

    [[nodiscard]] const Eigen::MatrixXd& process_noise() const { return process_noise_; }
    [[nodiscard]] const Eigen::MatrixXd& measurement_noise() const { return measurement_noise_; }
    [[nodiscard]] dynamics::DynamicsBase& dynamics() const { return *dyn_obj_; }

    /// Evaluate the measurement function at a given state.
    [[nodiscard]] Eigen::VectorXd estimate_measurement(const Eigen::VectorXd& state) const {
        if (h_) {
            return h_(state);
        }
        return get_measurement_matrix(state) * state;
    }

    /// Get the measurement Jacobian (constant or evaluated at state).
    [[nodiscard]] Eigen::MatrixXd get_measurement_matrix(const Eigen::VectorXd& state) const {
        if (auto* mat = std::get_if<Eigen::MatrixXd>(&H_)) {
            return *mat;
        }
        return std::get<std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>>(H_)(state);
    }

    // ---- Abstract interface ----

    struct CorrectionResult {
        T distribution;
        double likelihood;
    };

    [[nodiscard]] virtual T predict(double dt, const T& prev) const = 0;

    [[nodiscard]] virtual CorrectionResult correct(
        const Eigen::VectorXd& measurement,
        const T& predicted) const = 0;

    [[nodiscard]] virtual double gate(
        const Eigen::VectorXd& measurement,
        const T& predicted) const = 0;

protected:
    std::shared_ptr<dynamics::DynamicsBase> dyn_obj_;
    MeasurementFunc h_;
    MeasurementJacobian H_;
    Eigen::MatrixXd process_noise_;
    Eigen::MatrixXd measurement_noise_;
};

} // namespace brew::filters
