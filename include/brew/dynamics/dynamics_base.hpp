#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace brew::dynamics {

namespace detail {
    constexpr int scaled_dim(int d, int factor) {
        return d == Eigen::Dynamic ? Eigen::Dynamic : d * factor;
    }
}

template <typename Scalar = double, int D = Eigen::Dynamic>
class DynamicsBase {
public:
    using Vector = Eigen::Matrix<Scalar, D, 1>;
    using Matrix = Eigen::Matrix<Scalar, D, D>;
    using InputVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using InputMatrix = Eigen::Matrix<Scalar, D, Eigen::Dynamic>;
    using RotationMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ExtentMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    virtual ~DynamicsBase() = default;

    [[nodiscard]] virtual std::unique_ptr<DynamicsBase> clone() const = 0;

    [[nodiscard]] virtual std::vector<std::string> state_names() const = 0;

    [[nodiscard]] virtual Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const InputVector& input = InputVector{}) const = 0;

    [[nodiscard]] virtual Vector propagate_state(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& ,
        const InputVector& input = InputVector{}) const {
        return propagate_state(dt, state, input);
    }

    [[nodiscard]] virtual Matrix get_state_mat(
        Scalar dt,
        const Vector& state = Vector{}) const = 0;

    [[nodiscard]] virtual Matrix get_state_mat(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& ) const {
        return get_state_mat(dt, state);
    }

    [[nodiscard]] virtual InputMatrix get_input_mat(
        Scalar dt,
        const Vector& state = Vector{}) const = 0;

    [[nodiscard]] virtual InputMatrix get_input_mat(
        Scalar dt,
        const Vector& state,
        const RotationMatrix& ) const {
        return get_input_mat(dt, state);
    }

    [[nodiscard]] virtual ExtentMatrix propagate_extent(
        Scalar ,
        const Vector& ,
        const ExtentMatrix& extent) const {
        return extent;
    }

    [[nodiscard]] virtual bool is_lti() const { return false; }
};

}
