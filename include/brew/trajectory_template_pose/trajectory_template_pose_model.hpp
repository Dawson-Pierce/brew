#pragma once

#include "brew/shared/trajectory_window.hpp"
#include "brew/template_pose/template_pose_model.hpp"

namespace brew::models {

// @mex model
// @mex_name TrajectoryTemplatePose
// @mex_trajectory TemplatePose
template <typename Scalar = double, int D = Eigen::Dynamic>
class TrajectoryTemplatePose : public TrajectoryWindow<TemplatePose<Scalar, D>> {
    using Base = TrajectoryWindow<TemplatePose<Scalar, D>>;
public:
    TrajectoryTemplatePose() = default;
    using Base::Base;

    [[nodiscard]] std::unique_ptr<TrajectoryTemplatePose> clone() const {
        return std::make_unique<TrajectoryTemplatePose>(*this);
    }
    [[nodiscard]] std::unique_ptr<TrajectoryTemplatePose> clone_typed() const {
        return std::make_unique<TrajectoryTemplatePose>(*this);
    }
};

}
