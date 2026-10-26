#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace brew::fusion {

template <typename Trajectory>
bool moment_merge_trajectory(
    Trajectory& first,
    const Trajectory& second,
    double first_weight,
    double second_weight)
{
    const int live = first.stacked_size();
    const double total_weight = first_weight + second_weight;
    if (live <= 0 || live != second.stacked_size() || !(total_weight > 0.0)) {
        return false;
    }

    const Eigen::VectorXd first_mean = first.mean().head(live);
    const Eigen::VectorXd second_mean = second.mean().head(live);
    const Eigen::MatrixXd first_cov = first.covariance().topLeftCorner(live, live);
    const Eigen::MatrixXd second_cov = second.covariance().topLeftCorner(live, live);
    const double q_first = first_weight / total_weight;
    const double q_second = second_weight / total_weight;

    const Eigen::VectorXd merged_mean = q_first * first_mean + q_second * second_mean;
    const Eigen::VectorXd first_offset = first_mean - merged_mean;
    const Eigen::VectorXd second_offset = second_mean - merged_mean;
    Eigen::MatrixXd merged_cov =
        q_first * (first_cov + first_offset * first_offset.transpose())
        + q_second * (second_cov + second_offset * second_offset.transpose());
    merged_cov = 0.5 * (merged_cov + merged_cov.transpose());

    first.mean().head(live) = merged_mean;
    first.covariance().topLeftCorner(live, live) = merged_cov;
    for (int i = 0; i < first.window_size(); ++i) {
        first.history_at(i).mean() = first.mean_at(i);
        first.history_at(i).covariance() = first.cov_at(i, i);
    }
    return true;
}

}
