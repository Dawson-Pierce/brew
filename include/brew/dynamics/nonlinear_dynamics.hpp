#pragma once

#include "brew/dynamics/dynamics_base.hpp"

namespace brew::dynamics {

template <typename Scalar = double, int D = Eigen::Dynamic>
class NonlinearDynamics : public DynamicsBase<Scalar, D> {

};

}
