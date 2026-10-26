#pragma once

// Shared ICP machinery used by template-matching filters (TmEkf, TrajectoryTmEkf).
// Owns the base ICP algorithm, the template library, and a cache of PCA-ICP
// wrappers keyed by template id. Produces the fixed-size IcpPseudoMeasurement
// consumed by the filters' correct() step.

#include "brew/template_matching/icp_base.hpp"
#include "brew/template_matching/pca_icp.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include "brew/template_matching/so3.hpp"
#include "brew/template_matching/template_library.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>

namespace brew::template_matching {

struct IcpPseudoMeasurement {
    Eigen::Vector3d t;
    Eigen::Matrix3d R;
    Eigen::Matrix3d R_ref;
    double log_likelihood;
    int iterations;

    [[nodiscard]] Eigen::Vector3d rotation_innovation() const {
        return so3::log(Eigen::Matrix3d(R * R_ref.transpose()));
    }

    [[nodiscard]] Eigen::Matrix3d apply_rotation_delta(
        const Eigen::Vector3d& delta_rot) const {
        return so3::exp(delta_rot) * R_ref;
    }
};

class TmIcpRunner {
public:
    TmIcpRunner() = default;

    void set_icp(std::shared_ptr<IcpBase> icp) {
        icp_ = std::move(icp);
        icp_cache_.clear();
    }

    void set_template_library(std::shared_ptr<TemplateLibrary> lib) {
        template_library_ = std::move(lib);
        icp_cache_.clear();
    }

    [[nodiscard]] bool has_library() const {
        return static_cast<bool>(template_library_);
    }

    [[nodiscard]] bool ready() const {
        return static_cast<bool>(template_library_) && static_cast<bool>(icp_);
    }

    [[nodiscard]] IcpPseudoMeasurement run(
        const PointCloud& meas,
        int template_id,
        const Eigen::Matrix3d& R_pred,
        const Eigen::Vector3d& t_pred,
        bool cold_start) const {

        auto entry = template_library_->get_entry(template_id);
        // A partial cloud's centroid need not coincide with the body origin.
        const Eigen::Vector3d t_init = cold_start
            ? Eigen::Vector3d(meas.points().rowwise().mean() - R_pred * entry.pca_centroid)
            : t_pred;

        IcpResult icp_result;
        if (cold_start) {
            const auto& pca_icp = get_or_build_pca_icp(template_id, entry);
            icp_result = pca_icp.align(*entry.cloud, meas, R_pred, t_init);
        } else {
            icp_result = icp_->align(*entry.cloud, meas, R_pred, t_init);
        }

        const Eigen::Vector3d bbox_min = meas.points().rowwise().minCoeff();
        const Eigen::Vector3d bbox_max = meas.points().rowwise().maxCoeff();
        const double cluster_diag = (bbox_max - bbox_min).norm();
        const double drift = (icp_result.translation - t_init).norm();
        const double divergence_threshold = std::max(5.0 * cluster_diag, 50.0);

        if (drift > divergence_threshold
            || !icp_result.translation.allFinite()
            || !icp_result.rotation.allFinite()
            || !std::isfinite(icp_result.log_likelihood)) {
            return {
                t_init,
                R_pred,
                R_pred,
                icp_->log_likelihood(*entry.cloud, meas, R_pred, t_init),
                icp_result.iterations
            };
        }

        const Eigen::Matrix3d R_out = icp_result.rotation;
        const Eigen::Matrix3d R_ref = cold_start ? R_out : R_pred;
        return {
            Eigen::Vector3d(icp_result.translation),
            R_out,
            R_ref,
            icp_result.log_likelihood,
            icp_result.iterations
        };
    }

    [[nodiscard]] Eigen::Matrix3d template_spread(
        int template_id,
        const Eigen::Matrix3d& R) const {
        const auto entry = template_library_->get_entry(template_id);
        const Eigen::MatrixXd centered = entry.cloud->points().colwise() - entry.pca_centroid;
        return R * (centered * centered.transpose()
            / static_cast<double>(entry.cloud->num_points())) * R.transpose();
    }

    [[nodiscard]] Eigen::Matrix3d pose_rotation_covariance(
        int template_id,
        const Eigen::Matrix3d& R,
        double predicted_range,
        double point_sigma_sq,
        double angular_pitch,
        int return_cap) const {

        const auto source = template_library_->get(template_id);
        const auto& points = source->points();
        const Eigen::Vector3d center = points.rowwise().mean();
        const Eigen::MatrixXd centered = points.colwise() - center;
        const Eigen::Matrix3d C = centered * centered.transpose()
            / static_cast<double>(source->num_points());
        const Eigen::Matrix3d information = C.trace() * Eigen::Matrix3d::Identity() - C;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(information);
        const double floor = std::numeric_limits<double>::epsilon()
            * std::max(1.0, information.trace());
        const Eigen::Vector3d inverse_axes = solver.eigenvalues()
            .cwiseMax(floor).cwiseInverse();
        const Eigen::Matrix3d body_base = solver.eigenvectors()
            * inverse_axes.asDiagonal() * solver.eigenvectors().transpose();

        const Eigen::Vector3d spans = points.rowwise().maxCoeff()
            - points.rowwise().minCoeff();
        const double largest = spans.maxCoeff();
        const double middle = spans.sum() - largest - spans.minCoeff();
        const double area = largest * middle;
        const double footprint = predicted_range * angular_pitch;
        const double expected_returns = footprint > 0.0
            ? std::clamp(area / (footprint * footprint), 1.0,
                         static_cast<double>(return_cap))
            : static_cast<double>(return_cap);
        return (point_sigma_sq / expected_returns) * R * body_base * R.transpose();
    }

    [[nodiscard]] double surface_log_likelihood(
        const PointCloud& meas,
        int template_id,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        double sigma_sq,
        double outlier_probability,
        double outlier_volume) const {

        const auto source = template_library_->get(template_id);
        const auto& points = source->points();
        const int N = source->num_points();
        if (N == 0 || meas.num_points() == 0 || !R.allFinite() || !t.allFinite()) {
            return -std::numeric_limits<double>::infinity();
        }

        const double log_normal = -1.5 * std::log(2.0 * M_PI * sigma_sq);
        const double log_inlier = std::log1p(-outlier_probability);
        const double log_outlier = outlier_probability > 0.0
            ? std::log(outlier_probability / outlier_volume)
            : -std::numeric_limits<double>::infinity();
        const double log_N = std::log(static_cast<double>(N));
        double score = 0.0;

        for (int m = 0; m < meas.num_points(); ++m) {
            double max_log = -std::numeric_limits<double>::infinity();
            double sum_exp = 0.0;
            for (int i = 0; i < N; ++i) {
                const double log_component = log_normal
                    - (meas.points().col(m) - R * points.col(i) - t).squaredNorm()
                        / (2.0 * sigma_sq);
                if (log_component > max_log) {
                    sum_exp = sum_exp * std::exp(max_log - log_component) + 1.0;
                    max_log = log_component;
                } else {
                    sum_exp += std::exp(log_component - max_log);
                }
            }
            const double log_surface = max_log + std::log(sum_exp) - log_N;
            const double a = log_inlier + log_surface;
            const double b = log_outlier;
            score += std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
        }
        return score;
    }

private:

    [[nodiscard]] const PcaIcp& get_or_build_pca_icp(
        int id, const TemplateLibrary::Entry& entry) const {
        auto it = icp_cache_.find(id);
        if (it != icp_cache_.end()) return *it->second;

        auto pca = std::make_shared<PcaIcp>(icp_->clone(), *entry.cloud);
        pca->set_pca(entry.pca_axes, entry.pca_centroid);
        icp_cache_[id] = pca;
        return *pca;
    }

    std::shared_ptr<IcpBase> icp_;
    std::shared_ptr<TemplateLibrary> template_library_;
    mutable std::unordered_map<int, std::shared_ptr<PcaIcp>> icp_cache_;
};

}
