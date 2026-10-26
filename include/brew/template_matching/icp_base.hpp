#pragma once

#include "brew/template_matching/point_cloud.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace brew::template_matching {

struct IcpResult {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double rmse = 0.0;
    double log_likelihood = -std::numeric_limits<double>::infinity();
    double inlier_ratio = 0.0;
    int iterations = 0;
    bool converged = false;
};

struct IcpParams {
    int max_iterations = 50;
    double tolerance = 1e-6;
    double max_correspondence_dist = std::numeric_limits<double>::max();
    double sigma_sq = 1.0;
    double trim_fraction = 1.0;
    double observability_threshold = 0.0;  // Zero preserves the ordinary least-squares update.
    bool use_target_centroid_init = false;
    bool reverse_correspondences = true;
};

class IcpBase {
public:
    virtual ~IcpBase() = default;

    virtual void set_params(IcpParams params) {
        if (params.max_iterations < 0 || !std::isfinite(params.tolerance) ||
            params.tolerance < 0.0 || !std::isfinite(params.max_correspondence_dist) ||
            params.max_correspondence_dist < 0.0 || !std::isfinite(params.sigma_sq) ||
            params.sigma_sq <= 0.0 || !std::isfinite(params.trim_fraction) ||
            params.trim_fraction <= 0.0 || params.trim_fraction > 1.0 ||
            !std::isfinite(params.observability_threshold) ||
            params.observability_threshold < 0.0 || params.observability_threshold >= 1.0) {
            throw std::invalid_argument("Invalid ICP parameters.");
        }
        params_ = std::move(params);
    }
    [[nodiscard]] const IcpParams& params() const { return params_; }

    [[nodiscard]] virtual IcpResult align(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R_init = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& t_init = Eigen::Vector3d::Zero()) const = 0;

    [[nodiscard]] virtual double log_likelihood(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t) const = 0;

    [[nodiscard]] virtual std::unique_ptr<IcpBase> clone() const = 0;

protected:
    IcpParams params_;

    [[nodiscard]] std::vector<std::pair<int, int>> find_reverse_correspondences(
        const PointCloud& source,
        const Eigen::MatrixXd& target,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        double max_dist) const {

        struct Match {
            int src_idx;
            int tgt_idx;
            double dist_sq;
        };

        const double max_dist_sq = max_dist * max_dist;
        std::vector<Match> matches;
        matches.reserve(static_cast<std::size_t>(target.cols()));
        for (int target_index = 0; target_index < target.cols(); ++target_index) {
            const Eigen::Vector3d local = rotation.transpose()
                * (target.col(target_index) - translation);
            auto [source_index, distance] = source.nearest(local);
            if (source_index >= 0 && distance <= max_dist_sq) {
                matches.push_back({source_index, target_index, distance});
            }
        }

        if (params_.trim_fraction < 1.0 && !matches.empty()) {
            std::sort(matches.begin(), matches.end(),
                [](const Match& a, const Match& b) { return a.dist_sq < b.dist_sq; });
            const int keep = std::max(1, static_cast<int>(
                std::ceil(matches.size() * params_.trim_fraction)));
            matches.resize(static_cast<std::size_t>(keep));
        }

        std::vector<std::pair<int, int>> correspondences;
        correspondences.reserve(matches.size());
        for (const Match& match : matches) {
            correspondences.emplace_back(match.src_idx, match.tgt_idx);
        }
        return correspondences;
    }

    [[nodiscard]] std::vector<std::pair<int, int>> find_correspondences(
        const Eigen::MatrixXd& source_transformed,
        const Eigen::MatrixXd& target,
        double max_dist) const {

        struct Match {
            int src_idx;
            int tgt_idx;
            double dist_sq;
        };

        const int n_src = static_cast<int>(source_transformed.cols());
        const int n_tgt = static_cast<int>(target.cols());
        const double max_dist_sq = max_dist * max_dist;

        std::vector<Match> matches;
        matches.reserve(n_src);
        if (n_tgt == 0) return {};

        const Eigen::RowVectorXd target_norms = target.colwise().squaredNorm();
        constexpr int batch_size = 256;
        for (int first = 0; first < n_src; first += batch_size) {
            const int count = std::min(batch_size, n_src - first);
            const auto source_batch = source_transformed.middleCols(first, count);
            Eigen::MatrixXd distances = -2.0 * source_batch.transpose() * target;
            distances.rowwise() += target_norms;
            distances.colwise() += source_batch.colwise().squaredNorm().transpose();

            for (int row = 0; row < count; ++row) {
                Eigen::Index best_j = -1;
                const double best_dist_sq = distances.row(row).minCoeff(&best_j);
                if (best_j >= 0 && best_dist_sq <= max_dist_sq) {
                    matches.push_back({first + row, static_cast<int>(best_j),
                                       std::max(0.0, best_dist_sq)});
                }
            }
        }

        if (params_.trim_fraction < 1.0 && !matches.empty()) {
            std::sort(matches.begin(), matches.end(),
                [](const Match& a, const Match& b) { return a.dist_sq < b.dist_sq; });
            const int keep = std::max(1, static_cast<int>(
                std::ceil(matches.size() * params_.trim_fraction)));
            matches.resize(keep);
        }

        std::vector<std::pair<int, int>> correspondences;
        correspondences.reserve(matches.size());
        for (const auto& m : matches) {
            correspondences.emplace_back(m.src_idx, m.tgt_idx);
        }
        return correspondences;
    }

    [[nodiscard]] double compute_log_likelihood(
        const Eigen::MatrixXd& source_transformed,
        const Eigen::MatrixXd& target,
        const std::vector<std::pair<int, int>>& correspondences) const {

        if (correspondences.empty()) return -std::numeric_limits<double>::infinity();

        const int N = static_cast<int>(correspondences.size());
        const int d = static_cast<int>(source_transformed.rows());

        double sum_sq = 0.0;
        for (const auto& [si, ti] : correspondences) {
            sum_sq += (target.col(ti) - source_transformed.col(si)).squaredNorm();
        }

        return -0.5 * N * d * std::log(2.0 * M_PI * params_.sigma_sq)
               - sum_sq / (2.0 * params_.sigma_sq);
    }
};

}
