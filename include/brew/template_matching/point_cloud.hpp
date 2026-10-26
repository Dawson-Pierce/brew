#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brew::template_matching {

class PointCloud {
public:
    PointCloud() = default;

    inline explicit PointCloud(Eigen::MatrixXd points)
        : points_(std::move(points)) {}

    PointCloud(Eigen::MatrixXd points, Eigen::MatrixXd normals)
        : points_(std::move(points)), supplied_normals_(std::move(normals)) {
        if (points_.rows() != 3 || supplied_normals_.rows() != 3
            || supplied_normals_.cols() != points_.cols()
            || !supplied_normals_.allFinite()
            || (supplied_normals_.colwise().norm().array() <= 0.0).any()) {
            throw std::invalid_argument("Surface normals must be finite nonzero 3-vectors for every point.");
        }
        supplied_normals_.colwise().normalize();
    }

    [[nodiscard]] inline std::unique_ptr<PointCloud> clone() const {
        return std::make_unique<PointCloud>(*this);
    }

    [[nodiscard]] const Eigen::MatrixXd& points() const { return points_; }
    [[nodiscard]] Eigen::MatrixXd& points() {
        supplied_normals_.resize(0, 0);
        cached_normals_.resize(0, 0);
        cached_normals_k_ = 0;
        kd_nodes_.clear();
        return points_;
    }

    [[nodiscard]] int num_points() const { return static_cast<int>(points_.cols()); }
    [[nodiscard]] int dim() const { return static_cast<int>(points_.rows()); }

    [[nodiscard]] const Eigen::MatrixXd& normals(int k) const {
        if (supplied_normals_.cols() == num_points() && num_points() > 0) {
            return supplied_normals_;
        }
        if (dim() != 3 || num_points() < 2) {
            throw std::invalid_argument("Normal estimation requires a 3xN cloud with N at least two.");
        }
        k = std::clamp(k, 2, num_points());
        if (cached_normals_k_ == k && cached_normals_.cols() == num_points()) {
            return cached_normals_;
        }

        ensure_kd_tree();
        cached_normals_.resize(3, num_points());
        for (int i = 0; i < num_points(); ++i) {
            const std::vector<int> indices = nearest_k(points_.col(i), k);

            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            for (int j = 0; j < k; ++j) centroid += points_.col(indices[j]);
            centroid /= static_cast<double>(k);

            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (int j = 0; j < k; ++j) {
                const Eigen::Vector3d difference = points_.col(indices[j]) - centroid;
                covariance.noalias() += difference * difference.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            cached_normals_.col(i) = solver.eigenvectors().col(0);
        }
        cached_normals_k_ = k;
        return cached_normals_;
    }

    [[nodiscard]] std::pair<int, double> nearest(const Eigen::Vector3d& query) const {
        if (dim() != 3 || num_points() == 0) return {-1, std::numeric_limits<double>::infinity()};
        ensure_kd_tree();
        int best_index = -1;
        double best_distance = std::numeric_limits<double>::infinity();
        nearest_node(0, query, best_index, best_distance);
        return {best_index, best_distance};
    }

private:
    struct KdNode {
        int point = -1;
        int left = -1;
        int right = -1;
        int axis = 0;
    };

    void ensure_kd_tree() const {
        if (!kd_nodes_.empty()) return;
        std::vector<int> indices(static_cast<std::size_t>(num_points()));
        std::iota(indices.begin(), indices.end(), 0);
        kd_nodes_.reserve(indices.size());
        build_kd_node(indices, 0, static_cast<int>(indices.size()), 0);
    }

    int build_kd_node(std::vector<int>& indices, int first, int last, int depth) const {
        if (first >= last) return -1;
        const int axis = depth % 3;
        const int middle = first + (last - first) / 2;
        std::nth_element(indices.begin() + first, indices.begin() + middle,
                         indices.begin() + last, [&](int a, int b) {
            return points_(axis, a) < points_(axis, b);
        });
        const int node = static_cast<int>(kd_nodes_.size());
        kd_nodes_.push_back({indices[middle], -1, -1, axis});
        const int left = build_kd_node(indices, first, middle, depth + 1);
        const int right = build_kd_node(indices, middle + 1, last, depth + 1);
        kd_nodes_[static_cast<std::size_t>(node)].left = left;
        kd_nodes_[static_cast<std::size_t>(node)].right = right;
        return node;
    }

    void nearest_node(int node_index, const Eigen::Vector3d& query,
                      int& best_index, double& best_distance) const {
        if (node_index < 0) return;
        const KdNode& node = kd_nodes_[static_cast<std::size_t>(node_index)];
        const double distance = (points_.col(node.point) - query).squaredNorm();
        if (distance < best_distance) {
            best_distance = distance;
            best_index = node.point;
        }

        const double offset = query(node.axis) - points_(node.axis, node.point);
        const int near_node = offset < 0.0 ? node.left : node.right;
        const int far_node = offset < 0.0 ? node.right : node.left;
        nearest_node(near_node, query, best_index, best_distance);
        if (offset * offset < best_distance) {
            nearest_node(far_node, query, best_index, best_distance);
        }
    }

    [[nodiscard]] std::vector<int> nearest_k(const Eigen::Vector3d& query, int k) const {
        std::priority_queue<std::pair<double, int>> nearest;
        nearest_k_node(0, query, k, nearest);
        std::vector<int> result;
        result.reserve(nearest.size());
        while (!nearest.empty()) {
            result.push_back(nearest.top().second);
            nearest.pop();
        }
        return result;
    }

    void nearest_k_node(
        int node_index, const Eigen::Vector3d& query, int k,
        std::priority_queue<std::pair<double, int>>& nearest) const {
        if (node_index < 0) return;
        const KdNode& node = kd_nodes_[static_cast<std::size_t>(node_index)];
        const double distance = (points_.col(node.point) - query).squaredNorm();
        if (static_cast<int>(nearest.size()) < k) {
            nearest.emplace(distance, node.point);
        } else if (distance < nearest.top().first) {
            nearest.pop();
            nearest.emplace(distance, node.point);
        }

        const double offset = query(node.axis) - points_(node.axis, node.point);
        const int near_node = offset < 0.0 ? node.left : node.right;
        const int far_node = offset < 0.0 ? node.right : node.left;
        nearest_k_node(near_node, query, k, nearest);
        if (static_cast<int>(nearest.size()) < k || offset * offset < nearest.top().first) {
            nearest_k_node(far_node, query, k, nearest);
        }
    }

    Eigen::MatrixXd points_;
    Eigen::MatrixXd supplied_normals_;
    mutable Eigen::MatrixXd cached_normals_;
    mutable int cached_normals_k_ = 0;
    mutable std::vector<KdNode> kd_nodes_;
};

}
