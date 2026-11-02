#pragma once

#include "brew/models/trajectory_window.hpp"
#include "brew/models/ggiw.hpp"

namespace brew::models {

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

}
