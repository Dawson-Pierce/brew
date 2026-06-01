#pragma once

#include "brew/shared/trajectory_window.hpp"
#include "brew/ggiw/ggiw_model.hpp"

namespace brew::models {

/// Concrete GGIW trajectory model: a windowed trajectory of GGIW states.
/// Its own first-class type (not a Trajectory<T> instantiation); the windowed
/// ring-buffer mechanics are inherited from TrajectoryWindow.
// @mex model
// @mex_name TrajectoryGGIW
// @mex_trajectory GGIW
template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class TrajectoryGGIW : public TrajectoryWindow<GGIW<Scalar, D, De>> {
    using Base = TrajectoryWindow<GGIW<Scalar, D, De>>;
public:
    TrajectoryGGIW() = default;
    using Base::Base;

    [[nodiscard]] std::unique_ptr<TrajectoryGGIW> clone() const {
        return std::make_unique<TrajectoryGGIW>(*this);
    }
    [[nodiscard]] std::unique_ptr<TrajectoryGGIW> clone_typed() const {
        return std::make_unique<TrajectoryGGIW>(*this);
    }
};

} // namespace brew::models
