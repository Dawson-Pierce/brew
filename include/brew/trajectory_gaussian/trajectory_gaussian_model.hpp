#pragma once

#include "brew/shared/trajectory_window.hpp"
#include "brew/gaussian/gaussian_model.hpp"

namespace brew::models {

/// Concrete Gaussian trajectory model: a windowed trajectory of Gaussian states.
/// Its own first-class type (not a Trajectory<T> instantiation); the windowed
/// ring-buffer mechanics are inherited from TrajectoryWindow.
// @mex model
// @mex_name TrajectoryGaussian
// @mex_trajectory Gaussian
template <typename Scalar = double, int D = Eigen::Dynamic>
class TrajectoryGaussian : public TrajectoryWindow<Gaussian<Scalar, D>> {
    using Base = TrajectoryWindow<Gaussian<Scalar, D>>;
public:
    TrajectoryGaussian() = default;
    using Base::Base;

    [[nodiscard]] std::unique_ptr<TrajectoryGaussian> clone() const {
        return std::make_unique<TrajectoryGaussian>(*this);
    }
    [[nodiscard]] std::unique_ptr<TrajectoryGaussian> clone_typed() const {
        return std::make_unique<TrajectoryGaussian>(*this);
    }
};

} // namespace brew::models
