#include "brew/template_matching/point_cloud_io.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>

namespace brew::template_matching {

PointCloud load_point_cloud(const std::string& filepath) {
    YAML::Node j = YAML::LoadFile(filepath);

    const int dim = j["dim"].as<int>();
    const int num_points = j["num_points"].as<int>();
    const auto& pts = j["points"];

    Eigen::MatrixXd points(dim, num_points);
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dim; ++d) {
            points(d, i) = pts[i][d].as<double>();
        }
    }

    return PointCloud(std::move(points));
}

void save_point_cloud(const PointCloud& cloud, const std::string& filepath) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "dim" << YAML::Value << cloud.dim();
    out << YAML::Key << "num_points" << YAML::Value << cloud.num_points();
    out << YAML::Key << "points" << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < cloud.num_points(); ++i) {
        out << YAML::Flow << YAML::BeginSeq;
        for (int d = 0; d < cloud.dim(); ++d) {
            out << cloud.points()(d, i);
        }
        out << YAML::EndSeq;
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << out.c_str();
}

} // namespace brew::template_matching
