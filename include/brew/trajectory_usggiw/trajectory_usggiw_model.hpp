#pragma once

#include "brew/shared/trajectory_window.hpp"
#include "brew/usggiw/usggiw_model.hpp"

namespace brew::models {

// @mex model
// @mex_name TrajectoryUSGGIW
// @mex_trajectory USGGIW
template <typename Scalar = double, int De = Eigen::Dynamic>
class TrajectoryUSGGIW : public TrajectoryWindow<USGGIW<Scalar, De>> {
    using Base = TrajectoryWindow<USGGIW<Scalar, De>>;
public:
    TrajectoryUSGGIW() = default;
    TrajectoryUSGGIW(int state_dim, const USGGIW<Scalar, De>& initial,
                     int max_window = Base::kDefaultWindow)
        : Base(state_dim, initial, max_window) {}

    [[nodiscard]] std::unique_ptr<TrajectoryUSGGIW> clone() const {
        return std::make_unique<TrajectoryUSGGIW>(*this);
    }
    [[nodiscard]] std::unique_ptr<TrajectoryUSGGIW> clone_typed() const {
        return std::make_unique<TrajectoryUSGGIW>(*this);
    }

    [[nodiscard]] Eigen::Matrix<Scalar, 3, Eigen::Dynamic> state_history_directions() const {
        const auto& states = this->state_history();
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic> result(3, static_cast<int>(states.size()));
        for (int i = 0; i < static_cast<int>(states.size()); ++i)
            result.col(i) = states[static_cast<std::size_t>(i)].direction();
        return result;
    }

    [[nodiscard]] Eigen::Matrix<Scalar, 2, Eigen::Dynamic> state_history_angular_velocities() const {
        const auto& states = this->state_history();
        Eigen::Matrix<Scalar, 2, Eigen::Dynamic> result(2, static_cast<int>(states.size()));
        for (int i = 0; i < static_cast<int>(states.size()); ++i)
            result.col(i) = states[static_cast<std::size_t>(i)].angular_velocity();
        return result;
    }

    [[nodiscard]] Eigen::Matrix<Scalar, 3, Eigen::Dynamic> window_directions() const {
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic> result(3, this->window_size());
        for (int i = 0; i < this->window_size(); ++i)
            result.col(i) = this->history_at(i).direction();
        return result;
    }

    [[nodiscard]] Eigen::Matrix<Scalar, 2, Eigen::Dynamic> window_angular_velocities() const {
        Eigen::Matrix<Scalar, 2, Eigen::Dynamic> result(2, this->window_size());
        for (int i = 0; i < this->window_size(); ++i)
            result.col(i) = this->history_at(i).angular_velocity();
        return result;
    }

};

}  // namespace brew::models
