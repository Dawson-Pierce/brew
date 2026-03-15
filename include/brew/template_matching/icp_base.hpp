#pragma once

#include "brew/template_matching/point_cloud.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <cmath>

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
    double trim_fraction = 1.0; ///< Fraction of best correspondences to keep (0, 1]
};

/// Abstract base for ICP variants.
class IcpBase {
public:
    virtual ~IcpBase() = default;

    void set_params(IcpParams params) { params_ = std::move(params); }
    [[nodiscard]] const IcpParams& params() const { return params_; }

    [[nodiscard]] virtual IcpResult align(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R_init = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d& t_init = Eigen::Vector3d::Zero()) const = 0;

    /// Compute log-likelihood of the alignment (stays in log space).
    [[nodiscard]] virtual double log_likelihood(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t) const = 0;

    [[nodiscard]] virtual std::unique_ptr<IcpBase> clone() const = 0;

protected:
    IcpParams params_;

    /// Brute-force nearest-neighbor correspondences with optional trimming.
    /// Returns vector of (source_idx, target_idx) pairs, sorted by distance,
    /// trimmed to the best trim_fraction of matches within max_dist.
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
        for (int i = 0; i < n_src; ++i) {
            double best_dist_sq = std::numeric_limits<double>::max();
            int best_j = -1;
            for (int j = 0; j < n_tgt; ++j) {
                double dist_sq = (source_transformed.col(i) - target.col(j)).squaredNorm();
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_j = j;
                }
            }
            if (best_j >= 0 && best_dist_sq <= max_dist_sq) {
                matches.push_back({i, best_j, best_dist_sq});
            }
        }

        // Trim to best fraction
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

    /// Compute mean Gaussian log-likelihood per correspondence.
    /// Returns log-likelihood (not exponentiated) to stay in log space.
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

        // Mean log-likelihood per correspondence
        return -0.5 * d * std::log(2.0 * M_PI * params_.sigma_sq)
               - sum_sq / (2.0 * N * params_.sigma_sq);
    }
};

} // namespace brew::template_matching
