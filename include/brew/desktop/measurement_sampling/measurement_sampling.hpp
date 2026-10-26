#pragma once

#include "brew/template_matching/point_cloud.hpp"
#include <string>

namespace brew::measurement_sampling {

using template_matching::PointCloud;

[[nodiscard]] PointCloud load_stl(const std::string& filepath);

[[nodiscard]] Eigen::MatrixXd load_stl_triangles(const std::string& filepath);

[[nodiscard]] PointCloud load_stl_vsp(const std::string& filepath);

[[nodiscard]] Eigen::MatrixXd load_stl_triangles_vsp(const std::string& filepath);

[[nodiscard]] PointCloud sample_stl(const std::string& filepath, int num_points);

[[nodiscard]] PointCloud sample_stl(const std::string& filepath, int num_points,
                                    Eigen::MatrixXd& normals_out);

[[nodiscard]] PointCloud sample_stl_vsp(const std::string& filepath, int num_points);

[[nodiscard]] PointCloud sample_stl_vsp(const std::string& filepath, int num_points,
                                        Eigen::MatrixXd& normals_out);

[[nodiscard]] PointCloud sample_pc(const PointCloud& cloud, int num_points);

[[nodiscard]] PointCloud sample_circle(double radius, int num_points);

[[nodiscard]] PointCloud sample_rectangle(double width, double height, int num_points);

[[nodiscard]] PointCloud sample_triangle(double base, double height, int num_points);

[[nodiscard]] PointCloud sample_sphere(double radius, int num_points);

[[nodiscard]] PointCloud sample_box(double lx, double ly, double lz, int num_points);

}
