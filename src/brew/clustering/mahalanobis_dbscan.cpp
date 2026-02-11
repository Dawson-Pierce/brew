#include "brew/clustering/mahalanobis_dbscan.hpp"
#include <algorithm>
#include <queue>

namespace brew::clustering {

MahalanobisDBSCAN::MahalanobisDBSCAN(double epsilon, int min_pts)
    : epsilon_(epsilon), min_pts_(min_pts) {}

std::vector<int> MahalanobisDBSCAN::run_dbscan(
    const Eigen::MatrixXd& Z,
    const Eigen::MatrixXd& S_inv) const {

    const int n = static_cast<int>(Z.cols());
    std::vector<int> labels(n, -2);
    int cluster_id = 0;

    auto mahal_dist = [&](int i, int j) -> double {
        Eigen::VectorXd diff = Z.col(i) - Z.col(j);
        return std::sqrt(diff.transpose() * S_inv * diff);
    };

    auto range_query = [&](int idx) -> std::vector<int> {
        std::vector<int> neighbors;
        for (int j = 0; j < n; ++j) {
            if (mahal_dist(idx, j) <= epsilon_) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    };

    for (int i = 0; i < n; ++i) {
        if (labels[i] != -2) continue;

        auto neighbors = range_query(i);
        if (static_cast<int>(neighbors.size()) < min_pts_) {
            labels[i] = -1;
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
            if (labels[q] == -1) labels[q] = cluster_id;
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

std::vector<Eigen::MatrixXd> MahalanobisDBSCAN::cluster(
    const Eigen::MatrixXd& Z,
    const Eigen::MatrixXd& S) const {
    if (Z.cols() == 0) return {};

    Eigen::MatrixXd S_inv = S.inverse();
    auto labels = run_dbscan(Z, S_inv);
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

std::vector<Eigen::VectorXd> MahalanobisDBSCAN::get_unclustered(
    const Eigen::MatrixXd& Z,
    const Eigen::MatrixXd& S) const {
    if (Z.cols() == 0) return {};

    Eigen::MatrixXd S_inv = S.inverse();
    auto labels = run_dbscan(Z, S_inv);
    std::vector<Eigen::VectorXd> noise;
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        if (labels[i] == -1) noise.push_back(Z.col(i));
    }
    return noise;
}

} // namespace brew::clustering
