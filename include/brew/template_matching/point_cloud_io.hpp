#pragma once

#include "brew/template_matching/point_cloud.hpp"
#include <string>

namespace brew::template_matching {

// ---- I/O ----

/// Load a point cloud from a JSON file.
[[nodiscard]] PointCloud load_point_cloud(const std::string& filepath);

/// Save a point cloud to a JSON file.
void save_point_cloud(const PointCloud& cloud, const std::string& filepath);

/// Load a point cloud from a binary or ASCII STL file.
/// Extracts unique vertices from the triangle mesh.
[[nodiscard]] PointCloud load_stl(const std::string& filepath);

/// Sample N points uniformly on the surface of an STL mesh.
/// Uses area-weighted triangle sampling with barycentric coordinates.
[[nodiscard]] PointCloud sample_stl(const std::string& filepath, int num_points);

/// Sample N points with per-point face normals.
/// normals_out is resized to 3×N, each column is the unit face normal for that point.
[[nodiscard]] PointCloud sample_stl(const std::string& filepath, int num_points,
                                    Eigen::MatrixXd& normals_out);

/// Load triangle vertices from an STL file for wireframe rendering.
/// Returns a 3×(3*num_triangles) matrix: columns [v0,v1,v2, v0,v1,v2, ...].
[[nodiscard]] Eigen::MatrixXd load_stl_triangles(const std::string& filepath);

// ---- Sampling ----

/// Poisson disk sample an existing point cloud down to approximately N points.
[[nodiscard]] PointCloud sample_pc(const PointCloud& cloud, int num_points);

/// Sample N points uniformly on a 2D circle boundary.
[[nodiscard]] PointCloud sample_circle(double radius, int num_points);

/// Sample N points uniformly on a 2D rectangle boundary.
[[nodiscard]] PointCloud sample_rectangle(double width, double height, int num_points);

/// Sample N points uniformly on a 2D triangle boundary.
[[nodiscard]] PointCloud sample_triangle(double base, double height, int num_points);

/// Sample N points on a 3D sphere surface via Poisson disk sampling.
[[nodiscard]] PointCloud sample_sphere(double radius, int num_points);

/// Sample N points on a 3D box surface via Poisson disk sampling.
[[nodiscard]] PointCloud sample_box(double lx, double ly, double lz, int num_points);

} // namespace brew::template_matching
