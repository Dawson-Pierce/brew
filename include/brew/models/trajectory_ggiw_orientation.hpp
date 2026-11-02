#pragma once

#include "brew/models/trajectory_window.hpp"
#include "brew/models/ggiw_orientation.hpp"

namespace brew::models {

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

}
