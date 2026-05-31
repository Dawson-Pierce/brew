#include "brew/advanced/clustering/dbscan.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <utility>

namespace brew::clustering {

DBSCAN::DBSCAN(double epsilon, int min_pts,
               std::vector<int> dims, std::vector<double> scales)
    : epsilon_(epsilon), min_pts_(min_pts),
      dims_(std::move(dims)), scales_(std::move(scales)) {}

std::vector<int> DBSCAN::run_dbscan(const Eigen::MatrixXd& Z) const {
    const int n = static_cast<int>(Z.cols());

    // Metric matrix: the selected rows (default: all rows), each optionally
    // scaled by sqrt(scales[d]) so the plain Euclidean norm below evaluates
    // sqrt(sum_d scales[d] * (z_i,d - z_j,d)^2) -- a weighted distance.
    const bool need_copy = !dims_.empty() || !scales_.empty();
    Eigen::MatrixXd Zsub;
    if (!dims_.empty()) {
        Zsub.resize(static_cast<Eigen::Index>(dims_.size()), Z.cols());
        for (std::size_t r = 0; r < dims_.size(); ++r) {
            Zsub.row(static_cast<Eigen::Index>(r)) = Z.row(dims_[r]);
        }
    } else if (need_copy) {
        Zsub = Z;
    }
    if (!scales_.empty()) {
        const Eigen::Index m = std::min<Eigen::Index>(
            Zsub.rows(), static_cast<Eigen::Index>(scales_.size()));
        for (Eigen::Index r = 0; r < m; ++r) {
            Zsub.row(r) *= std::sqrt(std::max(scales_[static_cast<std::size_t>(r)], 0.0));
        }
    }
    const Eigen::MatrixXd& Zm = need_copy ? Zsub : Z;

    std::vector<int> labels(n, -2); // -2 = unvisited, -1 = noise
    int cluster_id = 0;

    auto range_query = [&](int idx) -> std::vector<int> {
        std::vector<int> neighbors;
        for (int j = 0; j < n; ++j) {
            if ((Zm.col(idx) - Zm.col(j)).norm() <= epsilon_) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    };

    for (int i = 0; i < n; ++i) {
        if (labels[i] != -2) continue;

        auto neighbors = range_query(i);
        if (static_cast<int>(neighbors.size()) < min_pts_) {
            labels[i] = -1; // noise
            continue;
        }

        labels[i] = cluster_id;
        std::queue<int> seed_set;
        for (int idx : neighbors) {
            if (idx != i) seed_set.push(idx);
        }

        while (!seed_set.empty()) {
            int q = seed_set.front();
            seed_set.pop();

            if (labels[q] == -1) {
                labels[q] = cluster_id;
            }
            if (labels[q] != -2) continue;

            labels[q] = cluster_id;
            auto q_neighbors = range_query(q);
            if (static_cast<int>(q_neighbors.size()) >= min_pts_) {
                for (int idx : q_neighbors) {
                    if (labels[idx] == -2 || labels[idx] == -1) {
                        seed_set.push(idx);
                    }
                }
            }
        }
        ++cluster_id;
    }
    return labels;
}

std::vector<Eigen::MatrixXd> DBSCAN::cluster(const Eigen::MatrixXd& Z) const {
    if (Z.cols() == 0) return {};

    auto labels = run_dbscan(Z);
    int max_label = *std::max_element(labels.begin(), labels.end());

    std::vector<Eigen::MatrixXd> clusters;
    for (int c = 0; c <= max_label; ++c) {
        std::vector<int> indices;
        for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
            if (labels[i] == c) indices.push_back(i);
        }
        if (indices.empty()) continue;

        Eigen::MatrixXd cluster_mat(Z.rows(), static_cast<int>(indices.size()));
        for (int j = 0; j < static_cast<int>(indices.size()); ++j) {
            cluster_mat.col(j) = Z.col(indices[j]);
        }
        clusters.push_back(std::move(cluster_mat));
    }
    return clusters;
}

std::vector<Eigen::VectorXd> DBSCAN::get_unclustered(const Eigen::MatrixXd& Z) const {
    if (Z.cols() == 0) return {};

    auto labels = run_dbscan(Z);
    std::vector<Eigen::VectorXd> noise;
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        if (labels[i] == -1) {
            noise.push_back(Z.col(i));
        }
    }
    return noise;
}

} // namespace brew::clustering
