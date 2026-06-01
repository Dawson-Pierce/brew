#pragma once

#include "brew/shared/trajectory_window.hpp"
#include "brew/ggiw_orientation/ggiw_orientation_model.hpp"

namespace brew::models {

/// Concrete GGIWOrientation trajectory model: a windowed trajectory of GGIWOrientation states.
/// Its own first-class type (not a Trajectory<T> instantiation); the windowed
/// ring-buffer mechanics are inherited from TrajectoryWindow.
// @mex model
// @mex_name TrajectoryGGIWOrientation
// @mex_trajectory GGIWOrientation
template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class TrajectoryGGIWOrientation : public TrajectoryWindow<GGIWOrientation<Scalar, D, De>> {
    using Base = TrajectoryWindow<GGIWOrientation<Scalar, D, De>>;
public:
    TrajectoryGGIWOrientation() = default;
    using Base::Base;

    [[nodiscard]] std::unique_ptr<TrajectoryGGIWOrientation> clone() const {
        return std::make_unique<TrajectoryGGIWOrientation>(*this);
    }
    [[nodiscard]] std::unique_ptr<TrajectoryGGIWOrientation> clone_typed() const {
        return std::make_unique<TrajectoryGGIWOrientation>(*this);
    }
};

} // namespace brew::models
