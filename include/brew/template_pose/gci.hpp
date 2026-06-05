#pragma once

// Generalized Covariance Intersection fusion for the template_pose package.
// The pose state is augmented: covariance() is (n_trans + 3) x (n_trans + 3) over
// [translation ; so(3) rotation tangent], with the rotation mean carried as the
// matrix rotation(). GCI is covariance intersection on this augmented tangent state,
// iterated on the SO(3) manifold: per iteration each source's residual is taken in
// the tangent at the current reference (translation difference + log(R_ref^T R_i)),
// combined in information form, and the reference advanced (t += dt, R = R*exp(dphi)).
// The tuple log-weight is the Gaussian geometric-mean normalization of the augmented
// residuals; template_id/pos_indices come from the first source. (If covariance() is
// only n_trans x n_trans the rotation is treated as a covariance-free Karcher mean.)
// Mixtures fuse combinatorially.
#include "brew/shared/mixture.hpp"
#include "brew/shared/fusion/gci_detail.hpp"
#include "brew/template_pose/template_pose_model.hpp"
#include "brew/template_matching/so3.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

namespace brew::template_pose {

template <typename Scalar = double, int D = Eigen::Dynamic, int N = Eigen::Dynamic>
models::Mixture<models::TemplatePose<Scalar, D>, N> gci(
    const std::vector<const models::Mixture<models::TemplatePose<Scalar, D>, N>*>& sources,
    std::vector<double> weights = {},
    double prune_threshold = 0.0) {
    using TP = models::TemplatePose<Scalar, D>;
    namespace fd = fusion::detail;
    namespace so3 = template_matching::so3;
    models::Mixture<TP, N> out;

    std::vector<const models::Mixture<TP, N>*> src;
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

    struct Fused { Eigen::VectorXd t; Eigen::MatrixXd P; Eigen::Matrix3d R; int tid; std::vector<int> pos; double logw; };
    std::vector<Fused> fused;
    const double tiny = 1e-300;

    while (true) {
        const int nt = static_cast<int>(src[0]->component(idx[0]).mean().size());
        const bool aug = (src[0]->component(idx[0]).covariance().rows() == nt + 3);
        const int na = aug ? nt + 3 : nt;

        Eigen::VectorXd t_ref = src[0]->component(idx[0]).mean().template cast<double>();
        Eigen::Matrix3d R_ref = src[0]->component(idx[0]).rotation().template cast<double>();
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Identity(na, na);

        for (int iter = 0; iter < 12; ++iter) {
            Lambda.setZero();
            Eigen::VectorXd eta = Eigen::VectorXd::Zero(na);
            for (std::size_t i = 0; i < M; ++i) {
                const TP& g = src[i]->component(idx[i]);
                Eigen::VectorXd xi(na);
                xi.head(nt) = g.mean().template cast<double>() - t_ref;
                if (aug) xi.tail(3) = so3::log(R_ref.transpose() * g.rotation().template cast<double>());
                const Eigen::MatrixXd Pinv = g.covariance().template cast<double>().inverse();
                Lambda += w[i] * Pinv;
                eta += w[i] * (Pinv * xi);
            }
            const Eigen::VectorXd xi_f = Lambda.ldlt().solve(eta);
            t_ref += xi_f.head(nt);
            if (aug) R_ref = R_ref * so3::exp(xi_f.tail(3));
            if (xi_f.norm() < 1e-12) break;
        }
        if (!aug) {
            std::vector<Eigen::Matrix3d> rots(M);
            for (std::size_t i = 0; i < M; ++i)
                rots[i] = src[i]->component(idx[i]).rotation().template cast<double>();
            R_ref = so3::weighted_mean(rots, w);
        }
        const Eigen::MatrixXd Pf = Lambda.inverse();

        double logdet_sum = 0, quad_sum = 0, logw_sum = 0;
        for (std::size_t i = 0; i < M; ++i) {
            const TP& g = src[i]->component(idx[i]);
            Eigen::VectorXd xi(na);
            xi.head(nt) = g.mean().template cast<double>() - t_ref;
            if (aug) xi.tail(3) = so3::log(R_ref.transpose() * g.rotation().template cast<double>());
            const Eigen::MatrixXd Pi = g.covariance().template cast<double>();
            logdet_sum += w[i] * fd::log_det_chol(Pi);
            quad_sum += w[i] * xi.dot(Pi.ldlt().solve(xi));
            logw_sum += w[i] * std::log(std::max(src[i]->weight(idx[i]), tiny));
        }
        const double log_c = 0.5 * fd::log_det_chol(Pf) - 0.5 * logdet_sum - 0.5 * quad_sum;
        const TP& ref = src[0]->component(idx[0]);
        fused.push_back({t_ref, Pf, R_ref, ref.template_id(), ref.pos_indices(), logw_sum + log_c});

        std::size_t j = M;
        while (j-- > 0) {
            if (++idx[j] < n[j]) break;
            idx[j] = 0;
        }
        if (j == static_cast<std::size_t>(-1)) break;
    }

    double maxlog = -std::numeric_limits<double>::infinity();
    for (const auto& f : fused) maxlog = std::max(maxlog, f.logw);
    double norm = 0.0;
    for (const auto& f : fused) norm += std::exp(f.logw - maxlog);
    for (const auto& f : fused) {
        const double weight = std::exp(f.logw - maxlog) / norm;
        if (weight >= prune_threshold) {
            out.add_component(std::make_unique<TP>(
                f.t.template cast<Scalar>(), f.P.template cast<Scalar>(),
                f.tid, f.pos, f.R.template cast<Scalar>()), weight);
        }
    }
    return out;
}

}  // namespace brew::template_pose
