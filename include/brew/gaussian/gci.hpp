#pragma once

// Generalized Covariance Intersection (geometric-mean / Chernoff) fusion for the
// gaussian package: f ∝ Π_i f_i^{w_i}. Each cross-source component tuple fuses by
// covariance intersection (P_f^{-1} = Σ w_i P_i^{-1}); the tuple's fused weight is
// Π_i w_{i,k}^{w_i} times the Gaussian geometric-mean normalization constant.
// Mixtures fuse combinatorially (Π_i n_i components) — prune/cap the result.
#include "brew/shared/mixture.hpp"
#include "brew/gaussian/gaussian_model.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace brew::gaussian {

namespace detail {
template <typename Mat>
inline double gci_log_det(const Mat& P) {
    Eigen::LLT<Mat> llt(P);
    double s = 0.0;
    for (Eigen::Index i = 0; i < P.rows(); ++i) s += std::log(llt.matrixL()(i, i));
    return 2.0 * s;
}
}  // namespace detail

template <typename Scalar = double, int D = Eigen::Dynamic, int N = Eigen::Dynamic>
models::Mixture<models::Gaussian<Scalar, D>, N> gci(
    const std::vector<const models::Mixture<models::Gaussian<Scalar, D>, N>*>& sources,
    std::vector<double> weights = {},
    double prune_threshold = 0.0) {
    using G = models::Gaussian<Scalar, D>;
    using Matrix = typename G::Matrix;
    using Vector = typename G::Vector;
    models::Mixture<G, N> out;

    std::vector<const models::Mixture<G, N>*> src;
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

    std::vector<Vector> means;
    std::vector<Matrix> covs;
    std::vector<double> logw;
    const double tiny = 1e-300;

    while (true) {
        const Eigen::Index d = src[0]->component(idx[0]).mean().size();
        Matrix Pinv_sum = Matrix::Zero(d, d);
        Vector info_sum = Vector::Zero(d);
        double quad_sum = 0.0, logdet_sum = 0.0, logw_sum = 0.0;

        for (std::size_t i = 0; i < M; ++i) {
            const G& g = src[i]->component(idx[i]);
            const Matrix Pinv = g.covariance().inverse();
            const Vector Pinv_m = Pinv * g.mean();
            Pinv_sum += w[i] * Pinv;
            info_sum += w[i] * Pinv_m;
            quad_sum += w[i] * g.mean().dot(Pinv_m);
            logdet_sum += w[i] * detail::gci_log_det(g.covariance());
            logw_sum += w[i] * std::log(std::max(src[i]->weight(idx[i]), tiny));
        }
        const Matrix Pf = Pinv_sum.inverse();
        const Vector mf = Pf * info_sum;
        const double mf_quad = mf.dot(Pinv_sum * mf);
        const double log_c = 0.5 * detail::gci_log_det(Pf) - 0.5 * logdet_sum
                           - 0.5 * (quad_sum - mf_quad);
        means.push_back(mf);
        covs.push_back(Pf);
        logw.push_back(logw_sum + log_c);

        std::size_t j = M;
        while (j-- > 0) {
            if (++idx[j] < n[j]) break;
            idx[j] = 0;
        }
        if (j == static_cast<std::size_t>(-1)) break;
    }

    double maxlog = -std::numeric_limits<double>::infinity();
    for (double l : logw) maxlog = std::max(maxlog, l);
    double norm = 0.0;
    for (double l : logw) norm += std::exp(l - maxlog);
    for (std::size_t t = 0; t < means.size(); ++t) {
        const double weight = std::exp(logw[t] - maxlog) / norm;
        if (weight >= prune_threshold) {
            out.add_component(std::make_unique<G>(means[t], covs[t]), weight);
        }
    }
    return out;
}

}  // namespace brew::gaussian
