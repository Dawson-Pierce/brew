#include "brew/template_matching/point_cloud_io.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <numbers>
#include <set>

namespace brew::template_matching {

// ---- JSON I/O ----

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

// ---- STL parsing helpers ----

struct Triangle {
    Eigen::Vector3d v0, v1, v2;
    double area() const {
        return 0.5 * (v1 - v0).cross(v2 - v0).norm();
    }
};

static bool is_binary_stl(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    // Skip 80-byte header
    file.seekg(80, std::ios::beg);
    if (!file.good()) return false;

    // Read triangle count
    uint32_t num_triangles = 0;
    file.read(reinterpret_cast<char*>(&num_triangles), 4);
    if (!file.good()) return false;

    // Check if file size matches binary STL: 80 + 4 + num_triangles * 50
    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    auto expected_size = static_cast<std::streamoff>(84 + num_triangles * 50);

    return file_size == expected_size;
}

static std::vector<Triangle> parse_binary_stl(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open STL file: " + filepath);
    }

    // Skip 80-byte header
    file.seekg(80, std::ios::beg);

    uint32_t num_triangles = 0;
    file.read(reinterpret_cast<char*>(&num_triangles), 4);

    std::vector<Triangle> triangles(num_triangles);
    for (uint32_t i = 0; i < num_triangles; ++i) {
        float buf[12];
        file.read(reinterpret_cast<char*>(buf), 48);
        // buf[0..2] = normal (skip), buf[3..5] = v0, buf[6..8] = v1, buf[9..11] = v2
        triangles[i].v0 = Eigen::Vector3d(buf[3], buf[4], buf[5]);
        triangles[i].v1 = Eigen::Vector3d(buf[6], buf[7], buf[8]);
        triangles[i].v2 = Eigen::Vector3d(buf[9], buf[10], buf[11]);

        uint16_t attr;
        file.read(reinterpret_cast<char*>(&attr), 2);
    }

    return triangles;
}

static std::vector<Triangle> parse_ascii_stl(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open STL file: " + filepath);
    }

    std::vector<Triangle> triangles;
    std::string line;
    Triangle current;
    int vertex_idx = 0;

    while (std::getline(file, line)) {
        // Trim leading whitespace
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.rfind("vertex ", 0) == 0) {
            double x, y, z;
            if (std::sscanf(line.c_str(), "vertex %lf %lf %lf", &x, &y, &z) == 3) {
                Eigen::Vector3d v(x, y, z);
                if (vertex_idx == 0) current.v0 = v;
                else if (vertex_idx == 1) current.v1 = v;
                else if (vertex_idx == 2) {
                    current.v2 = v;
                    triangles.push_back(current);
                }
                vertex_idx = (vertex_idx + 1) % 3;
            }
        }
    }

    return triangles;
}

static std::vector<Triangle> parse_stl(const std::string& filepath) {
    if (is_binary_stl(filepath)) {
        return parse_binary_stl(filepath);
    }
    return parse_ascii_stl(filepath);
}

// Extract unique vertices from triangles
static Eigen::MatrixXd extract_unique_vertices(const std::vector<Triangle>& triangles) {
    // Use a set-like approach with snapping to remove near-duplicate vertices
    struct Vec3Compare {
        bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
            constexpr double eps = 1e-10;
            if (std::abs(a(0) - b(0)) > eps) return a(0) < b(0);
            if (std::abs(a(1) - b(1)) > eps) return a(1) < b(1);
            return a(2) < b(2) - eps;
        }
    };

    std::set<Eigen::Vector3d, Vec3Compare> unique;
    for (const auto& tri : triangles) {
        unique.insert(tri.v0);
        unique.insert(tri.v1);
        unique.insert(tri.v2);
    }

    Eigen::MatrixXd points(3, static_cast<int>(unique.size()));
    int i = 0;
    for (const auto& v : unique) {
        points.col(i++) = v;
    }
    return points;
}

// Sample a random point on a triangle using barycentric coordinates
static Eigen::Vector3d sample_triangle(const Triangle& tri, double u, double v) {
    // Map uniform [0,1]^2 to barycentric coordinates
    double su = std::sqrt(u);
    double b0 = 1.0 - su;
    double b1 = v * su;
    double b2 = 1.0 - b0 - b1;
    return b0 * tri.v0 + b1 * tri.v1 + b2 * tri.v2;
}

PointCloud load_stl(const std::string& filepath) {
    auto triangles = parse_stl(filepath);
    if (triangles.empty()) {
        throw std::runtime_error("No triangles found in STL file: " + filepath);
    }
    Eigen::MatrixXd points = extract_unique_vertices(triangles);
    return PointCloud(std::move(points));
}

// ---- Poisson disk sampling ----

static Eigen::MatrixXd poisson_disk_filter(const Eigen::MatrixXd& candidates, int num_points) {
    const int N_cand = static_cast<int>(candidates.cols());

    if (N_cand <= num_points) return candidates;

    std::vector<int> indices(N_cand);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<int> accepted;
    accepted.reserve(num_points);

    // Binary search for the right exclusion radius
    double r_low = 0.0;
    double r_high = 0.0;
    for (int i = 0; i < N_cand; ++i) {
        r_high = std::max(r_high, candidates.col(i).norm());
    }
    r_high *= 2.0 / std::sqrt(static_cast<double>(num_points));

    for (int iter = 0; iter < 30; ++iter) {
        double r = (r_low + r_high) / 2.0;
        double r_sq = r * r;

        accepted.clear();
        for (int idx : indices) {
            bool too_close = false;
            for (int acc : accepted) {
                if ((candidates.col(idx) - candidates.col(acc)).squaredNorm() < r_sq) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                accepted.push_back(idx);
            }
        }

        if (static_cast<int>(accepted.size()) == num_points) break;
        if (static_cast<int>(accepted.size()) < num_points) {
            r_high = r;
        } else {
            r_low = r;
        }
    }

    const int n_out = std::min(static_cast<int>(accepted.size()), num_points);
    Eigen::MatrixXd result(candidates.rows(), n_out);
    for (int i = 0; i < n_out; ++i) {
        result.col(i) = candidates.col(accepted[i]);
    }
    return result;
}

PointCloud sample_pc(const PointCloud& cloud, int num_points) {
    if (cloud.num_points() <= num_points) {
        return *cloud.clone();
    }
    Eigen::MatrixXd filtered = poisson_disk_filter(cloud.points(), num_points);
    return PointCloud(std::move(filtered));
}

// ---- 2D sampling primitives ----

PointCloud sample_circle(double radius, int num_points) {
    Eigen::MatrixXd points(2, num_points);
    for (int i = 0; i < num_points; ++i) {
        double theta = 2.0 * std::numbers::pi * i / num_points;
        points(0, i) = radius * std::cos(theta);
        points(1, i) = radius * std::sin(theta);
    }
    return PointCloud(std::move(points));
}

PointCloud sample_rectangle(double width, double height, int num_points) {
    const double perimeter = 2.0 * (width + height);
    const double spacing = perimeter / num_points;

    Eigen::MatrixXd points(2, num_points);
    double hw = width / 2.0;
    double hh = height / 2.0;

    for (int i = 0; i < num_points; ++i) {
        double s = i * spacing;
        if (s < width) {
            points(0, i) = -hw + s;
            points(1, i) = -hh;
        } else if (s < width + height) {
            points(0, i) = hw;
            points(1, i) = -hh + (s - width);
        } else if (s < 2.0 * width + height) {
            points(0, i) = hw - (s - width - height);
            points(1, i) = hh;
        } else {
            points(0, i) = -hw;
            points(1, i) = hh - (s - 2.0 * width - height);
        }
    }
    return PointCloud(std::move(points));
}

PointCloud sample_triangle(double base, double height, int num_points) {
    // Isosceles triangle centered at centroid
    Eigen::Vector2d v0(-base / 2.0, -height / 3.0);   // bottom-left
    Eigen::Vector2d v1( base / 2.0, -height / 3.0);   // bottom-right
    Eigen::Vector2d v2(        0.0, 2.0 * height / 3.0); // top

    double l01 = (v1 - v0).norm();
    double l12 = (v2 - v1).norm();
    double l20 = (v0 - v2).norm();
    double perimeter = l01 + l12 + l20;
    double spacing = perimeter / num_points;

    std::array<Eigen::Vector2d, 3> verts = {v0, v1, v2};
    std::array<double, 3> lengths = {l01, l12, l20};

    Eigen::MatrixXd points(2, num_points);
    int edge = 0;
    double edge_s = 0.0;
    for (int i = 0; i < num_points; ++i) {
        while (edge_s > lengths[edge] + 1e-12) {
            edge_s -= lengths[edge];
            edge = (edge + 1) % 3;
        }
        double t = edge_s / lengths[edge];
        points.col(i) = (1.0 - t) * verts[edge] + t * verts[(edge + 1) % 3];
        edge_s += spacing;
    }
    return PointCloud(std::move(points));
}

// ---- 3D sampling primitives ----

PointCloud sample_sphere(double radius, int num_points) {
    const int n_candidates = num_points * 5;
    Eigen::MatrixXd candidates(3, n_candidates);

    const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    for (int i = 0; i < n_candidates; ++i) {
        double theta = 2.0 * std::numbers::pi * i / golden_ratio;
        double phi = std::acos(1.0 - 2.0 * (i + 0.5) / n_candidates);
        candidates(0, i) = radius * std::sin(phi) * std::cos(theta);
        candidates(1, i) = radius * std::sin(phi) * std::sin(theta);
        candidates(2, i) = radius * std::cos(phi);
    }

    Eigen::MatrixXd filtered = poisson_disk_filter(candidates, num_points);
    return PointCloud(std::move(filtered));
}

PointCloud sample_box(double lx, double ly, double lz, int num_points) {
    const double a_xy = lx * ly;
    const double a_xz = lx * lz;
    const double a_yz = ly * lz;
    const double total_area = 2.0 * (a_xy + a_xz + a_yz);

    const int n_candidates = num_points * 5;
    Eigen::MatrixXd candidates(3, n_candidates);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const double hx = lx / 2.0;
    const double hy = ly / 2.0;
    const double hz = lz / 2.0;

    for (int i = 0; i < n_candidates; ++i) {
        double r = dist(rng) * total_area;
        double u = dist(rng);
        double v = dist(rng);

        if (r < a_yz) {
            candidates(0, i) = hx;
            candidates(1, i) = (u - 0.5) * ly;
            candidates(2, i) = (v - 0.5) * lz;
        } else if (r < 2.0 * a_yz) {
            candidates(0, i) = -hx;
            candidates(1, i) = (u - 0.5) * ly;
            candidates(2, i) = (v - 0.5) * lz;
        } else if (r < 2.0 * a_yz + a_xz) {
            candidates(0, i) = (u - 0.5) * lx;
            candidates(1, i) = hy;
            candidates(2, i) = (v - 0.5) * lz;
        } else if (r < 2.0 * a_yz + 2.0 * a_xz) {
            candidates(0, i) = (u - 0.5) * lx;
            candidates(1, i) = -hy;
            candidates(2, i) = (v - 0.5) * lz;
        } else if (r < 2.0 * a_yz + 2.0 * a_xz + a_xy) {
            candidates(0, i) = (u - 0.5) * lx;
            candidates(1, i) = (v - 0.5) * ly;
            candidates(2, i) = hz;
        } else {
            candidates(0, i) = (u - 0.5) * lx;
            candidates(1, i) = (v - 0.5) * ly;
            candidates(2, i) = -hz;
        }
    }

    Eigen::MatrixXd filtered = poisson_disk_filter(candidates, num_points);
    return PointCloud(std::move(filtered));
}

} // namespace brew::template_matching
