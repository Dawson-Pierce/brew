#include "brew/template_matching/point_cloud_io.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace brew::template_matching {

PointCloud load_point_cloud(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    nlohmann::json j;
    file >> j;

    const int dim = j.at("dim").get<int>();
    const int num_points = j.at("num_points").get<int>();
    const auto& pts = j.at("points");

    Eigen::MatrixXd points(dim, num_points);
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dim; ++d) {
            points(d, i) = pts[i][d].get<double>();
        }
    }

    return PointCloud(std::move(points));
}

void save_point_cloud(const PointCloud& cloud, const std::string& filepath) {
    nlohmann::json j;
    j["dim"] = cloud.dim();
    j["num_points"] = cloud.num_points();

    auto& pts = j["points"];
    pts = nlohmann::json::array();
    for (int i = 0; i < cloud.num_points(); ++i) {
        nlohmann::json pt = nlohmann::json::array();
        for (int d = 0; d < cloud.dim(); ++d) {
            pt.push_back(cloud.points()(d, i));
        }
        pts.push_back(std::move(pt));
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << j.dump(2);
}

} // namespace brew::template_matching
