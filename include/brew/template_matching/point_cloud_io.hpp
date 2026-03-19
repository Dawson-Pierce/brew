#pragma once

#include "brew/template_matching/point_cloud.hpp"
#include <string>

namespace brew::template_matching {

/// Load a point cloud from a YAML file.
[[nodiscard]] PointCloud load_point_cloud(const std::string& filepath);

/// Save a point cloud to a YAML file.
void save_point_cloud(const PointCloud& cloud, const std::string& filepath);

} // namespace brew::template_matching
