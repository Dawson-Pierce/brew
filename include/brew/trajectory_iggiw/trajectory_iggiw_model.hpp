#pragma once

#include "brew/shared/trajectory_window.hpp"
#include "brew/iggiw/iggiw_model.hpp"

namespace brew::models {

/// Concrete IGGIW trajectory model: a windowed trajectory of IGGIW states.
/// Its own first-class type (not a Trajectory<T> instantiation); the windowed
/// ring-buffer mechanics are inherited from TrajectoryWindow.
// @mex model
// @mex_name TrajectoryIGGIW
// @mex_trajectory IGGIW
template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic>
class TrajectoryIGGIW : public TrajectoryWindow<IGGIW<Scalar, D, De>> {
    using Base = TrajectoryWindow<IGGIW<Scalar, D, De>>;
public:
    TrajectoryIGGIW() = default;
    using Base::Base;

    [[nodiscard]] std::unique_ptr<TrajectoryIGGIW> clone() const {
        return std::make_unique<TrajectoryIGGIW>(*this);
    }
    [[nodiscard]] std::unique_ptr<TrajectoryIGGIW> clone_typed() const {
        return std::make_unique<TrajectoryIGGIW>(*this);
    }
};

} // namespace brew::models
