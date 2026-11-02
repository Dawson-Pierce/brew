#pragma once

#include "brew/models/trajectory_window.hpp"
#include "brew/models/gaussian.hpp"

namespace brew::models {

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

}
