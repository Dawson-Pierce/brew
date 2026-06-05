#pragma once

// Model-agnostic numeric helpers for Generalized Covariance Intersection
// (geometric-mean) fusion weights.
#include <Eigen/Dense>
#include <cmath>

namespace brew::fusion::detail {

template <typename Mat>
inline double log_det_chol(const Mat& M) {
    Eigen::LLT<Mat> llt(M);
    double s = 0.0;
    for (Eigen::Index i = 0; i < M.rows(); ++i) s += std::log(llt.matrixL()(i, i));
    return 2.0 * s;
}

// log of the multivariate gamma Gamma_p(a).
inline double log_mv_gamma(double a, int p) {
    double r = 0.25 * p * (p - 1) * std::log(M_PI);
    for (int j = 1; j <= p; ++j) r += std::lgamma(a + 0.5 * (1.0 - j));
    return r;
}

}  // namespace brew::fusion::detail
