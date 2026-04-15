#pragma once

#include <Eigen/Dense>
#include <memory>

namespace brew::template_matching {

/// Simple 2D/3D point cloud container. Columns are points.
class PointCloud {
public:
    PointCloud() = default;

    inline explicit PointCloud(Eigen::MatrixXd points)
        : points_(std::move(points)) {}

    [[nodiscard]] inline std::unique_ptr<PointCloud> clone() const {
        return std::make_unique<PointCloud>(points_);
    }

    [[nodiscard]] const Eigen::MatrixXd& points() const { return points_; }
    [[nodiscard]] Eigen::MatrixXd& points() { return points_; }

    [[nodiscard]] int num_points() const { return static_cast<int>(points_.cols()); }
    [[nodiscard]] int dim() const { return static_cast<int>(points_.rows()); }

private:
    Eigen::MatrixXd points_; // d×N matrix (d = 2 or 3)
};

} // namespace brew::template_matching
