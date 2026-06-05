#pragma once

// Generalized Covariance Intersection fusion for the iggiw package. IGGIW shares
// the Gamma-Gaussian-Inverse-Wishart structure of GGIW, so each cross-source tuple
// pools Gamma (alpha,beta), Gaussian (covariance intersection) and IW (v,V) by
// weighted natural parameters, with the geometric-mean normalization constant of
// all three as its log-weight (Granstrom IW convention). Mixtures fuse combinatorially.
#include "brew/shared/mixture.hpp"
#include "brew/shared/fusion/gci_detail.hpp"
#include "brew/iggiw/iggiw_model.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

namespace brew::iggiw {

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic, int N = Eigen::Dynamic>
models::Mixture<models::IGGIW<Scalar, D, De>, N> gci(
    const std::vector<const models::Mixture<models::IGGIW<Scalar, D, De>, N>*>& sources,
    std::vector<double> weights = {},
    double prune_threshold = 0.0) {
    using GG = models::IGGIW<Scalar, D, De>;
    using Vector = typename GG::Vector;
    using Matrix = typename GG::Matrix;
    using ExtentMatrix = typename GG::ExtentMatrix;
    namespace fd = fusion::detail;
    models::Mixture<GG, N> out;

    std::vector<const models::Mixture<GG, N>*> src;
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

    struct Fused { Scalar a, b, v; Vector m; Matrix P; ExtentMatrix V; double logw; };
    std::vector<Fused> fused;
    const double tiny = 1e-300;

    while (true) {
        const Eigen::Index ds = src[0]->component(idx[0]).mean().size();
        const int de = src[0]->component(idx[0]).extent_dim();
        Matrix Pinv_sum = Matrix::Zero(ds, ds);
        Vector info_sum = Vector::Zero(ds);
        ExtentMatrix Vf = ExtentMatrix::Zero(de, de);
        double quad_sum = 0, logdet_P_sum = 0, logw_sum = 0;
        double af = 0, bf = 0, vf = 0, gamma_acc = 0, iw_acc = 0;

        for (std::size_t i = 0; i < M; ++i) {
            const GG& g = src[i]->component(idx[i]);
            const Matrix Pinv = g.covariance().inverse();
            const Vector Pinv_m = Pinv * g.mean();
            Pinv_sum += w[i] * Pinv;
            info_sum += w[i] * Pinv_m;
            quad_sum += w[i] * g.mean().dot(Pinv_m);
            logdet_P_sum += w[i] * fd::log_det_chol(g.covariance());

            af += w[i] * g.alpha();
            bf += w[i] * g.beta();
            gamma_acc += w[i] * (g.alpha() * std::log(g.beta()) - std::lgamma(g.alpha()));

            vf += w[i] * g.v();
            Vf += w[i] * g.V();
            const double ai = 0.5 * (g.v() - de - 1.0);
            iw_acc += w[i] * (ai * fd::log_det_chol(g.V()) - fd::log_mv_gamma(ai, de));

            logw_sum += w[i] * std::log(std::max(src[i]->weight(idx[i]), tiny));
        }
        const Matrix Pf = Pinv_sum.inverse();
        const Vector mf = Pf * info_sum;
        const double log_cG = 0.5 * fd::log_det_chol(Pf) - 0.5 * logdet_P_sum
                            - 0.5 * (quad_sum - mf.dot(Pinv_sum * mf));
        const double log_cGamma = gamma_acc + std::lgamma(af) - af * std::log(bf);
        const double af_iw = 0.5 * (vf - de - 1.0);
        const double log_cIW = iw_acc - af_iw * fd::log_det_chol(Vf) + fd::log_mv_gamma(af_iw, de);

        fused.push_back({static_cast<Scalar>(af), static_cast<Scalar>(bf), static_cast<Scalar>(vf),
                         mf, Pf, Vf, logw_sum + log_cG + log_cGamma + log_cIW});

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
            out.add_component(std::make_unique<GG>(f.a, f.b, f.m, f.P, f.v, f.V), weight);
        }
    }
    return out;
}

}  // namespace brew::iggiw
