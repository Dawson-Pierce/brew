#pragma once

// Generalized Covariance Intersection fusion for the trajectory_template_pose
// package. Covariance intersection on the stacked windowed kinematic Gaussian gives
// the tuple log-weight; each inner history step's SO(3) rotation fuses by a weighted
// geodesic (Karcher) mean on the manifold (template_id/pos_indices from the first
// source). Equal-window-size sources only.
#include "brew/shared/mixture.hpp"
#include "brew/shared/fusion/gci_detail.hpp"
#include "brew/trajectory_template_pose/trajectory_template_pose_model.hpp"
#include "brew/template_matching/so3.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace brew::trajectory_template_pose {

template <typename Scalar = double, int D = Eigen::Dynamic, int N = Eigen::Dynamic>
models::Mixture<models::TrajectoryTemplatePose<Scalar, D>, N> gci(
    const std::vector<const models::Mixture<models::TrajectoryTemplatePose<Scalar, D>, N>*>& sources,
    std::vector<double> weights = {},
    double prune_threshold = 0.0) {
    using TG = models::TrajectoryTemplatePose<Scalar, D>;
    using Inner = typename TG::InnerType;
    namespace fd = fusion::detail;
    models::Mixture<TG, N> out;

    std::vector<const models::Mixture<TG, N>*> src;
    std::vector<double> w;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (sources[i] && !sources[i]->empty()) {
            src.push_back(sources[i]);
            w.push_back(weights.size() == sources.size() ? weights[i] : 1.0);
        }
    }
    const std::size_t M = src.size();
    if (M == 0) return out;
    double wsum = 0.0;
    for (double x : w) wsum += x;
    for (double& x : w) x /= wsum;

    std::vector<std::size_t> idx(M, 0), n(M);
    for (std::size_t i = 0; i < M; ++i) n[i] = src[i]->size();

    struct Fused { std::unique_ptr<TG> traj; double logw; };
    std::vector<Fused> fused;
    const double tiny = 1e-300;

    while (true) {
        const int ss = src[0]->component(idx[0]).stacked_size();
        bool same = ss > 0;
        for (std::size_t i = 1; i < M && same; ++i)
            same = (src[i]->component(idx[i]).stacked_size() == ss);

        if (same) {
            Eigen::MatrixXd Pinv_sum = Eigen::MatrixXd::Zero(ss, ss);
            Eigen::VectorXd info_sum = Eigen::VectorXd::Zero(ss);
            double quad_sum = 0, logdet_sum = 0, logw_sum = 0;
            for (std::size_t i = 0; i < M; ++i) {
                const TG& g = src[i]->component(idx[i]);
                const Eigen::MatrixXd Ci = g.covariance().topLeftCorner(ss, ss).template cast<double>();
                const Eigen::VectorXd mi = g.mean().head(ss).template cast<double>();
                const Eigen::MatrixXd Cinv = Ci.inverse();
                const Eigen::VectorXd Cinv_m = Cinv * mi;
                Pinv_sum += w[i] * Cinv;
                info_sum += w[i] * Cinv_m;
                quad_sum += w[i] * mi.dot(Cinv_m);
                logdet_sum += w[i] * fd::log_det_chol(Ci);
                logw_sum += w[i] * std::log(std::max(src[i]->weight(idx[i]), tiny));
            }
            const Eigen::MatrixXd Pf = Pinv_sum.inverse();
            const Eigen::VectorXd mf = Pf * info_sum;
            const double log_cG = 0.5 * fd::log_det_chol(Pf) - 0.5 * logdet_sum
                                - 0.5 * (quad_sum - mf.dot(Pinv_sum * mf));

            auto traj = src[0]->component(idx[0]).clone_typed();
            traj->mean().head(ss) = mf.template cast<Scalar>();
            traj->covariance().topLeftCorner(ss, ss) = Pf.template cast<Scalar>();
            for (int s = 0; s < traj->window_size(); ++s) {
                typename Inner::Vector ms = traj->mean_at(s);
                typename Inner::Matrix Ps = traj->cov_at(s, s);
                std::vector<Eigen::Matrix3d> rots(M);
                for (std::size_t i = 0; i < M; ++i)
                    rots[i] = src[i]->component(idx[i]).history_at(s).rotation().template cast<double>();
                const typename Inner::RotationMatrix Rf =
                    template_matching::so3::weighted_mean(rots, w).template cast<Scalar>();
                const Inner& ref = src[0]->component(idx[0]).history_at(s);
                traj->history_at(s) = Inner(ms, Ps, ref.template_id(), ref.pos_indices(), Rf);
            }
            fused.push_back({std::move(traj), logw_sum + log_cG});
        }

        std::size_t j = M;
        while (j-- > 0) {
            if (++idx[j] < n[j]) break;
            idx[j] = 0;
        }
        if (j == static_cast<std::size_t>(-1)) break;
    }

    if (fused.empty()) return out;
    double maxlog = -std::numeric_limits<double>::infinity();
    for (const auto& f : fused) maxlog = std::max(maxlog, f.logw);
    double norm = 0.0;
    for (const auto& f : fused) norm += std::exp(f.logw - maxlog);
    for (auto& f : fused) {
        const double weight = std::exp(f.logw - maxlog) / norm;
        if (weight >= prune_threshold) out.add_component(std::move(f.traj), weight);
    }
    return out;
}

}  // namespace brew::trajectory_template_pose
