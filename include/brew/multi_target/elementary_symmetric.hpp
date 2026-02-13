#pragma once

#include <Eigen/Dense>
#include <cassert>

namespace brew::multi_target {

/// Compute elementary symmetric functions (ESFs) of a vector z.
/// Given z = [z_1, ..., z_m], computes e_0, e_1, ..., e_m where:
///   e_0 = 1
///   e_k = sum of all products of k distinct elements from z
/// Uses the efficient O(m^2) recurrence from Vo & Ma (2006).
/// Returns a vector of length m+1: [e_0, e_1, ..., e_m].
[[nodiscard]] inline Eigen::VectorXd elementary_symmetric_functions(
    const Eigen::VectorXd& z)
{
    const int m = static_cast<int>(z.size());
    Eigen::VectorXd esf = Eigen::VectorXd::Zero(m + 1);
    esf(0) = 1.0;

    for (int i = 0; i < m; ++i) {
        // Process in reverse to avoid overwriting values we still need
        for (int j = i + 1; j >= 1; --j) {
            esf(j) += z(i) * esf(j - 1);
        }
    }

    return esf;
}

/// Compute ESFs of z with the j-th element removed, using the "divide-out"
/// approach. Given the ESFs of the full vector z, computes the ESFs of
/// z \ {z_j} in O(m) time via the reverse recurrence:
///   esf_minus[0] = 1
///   esf_minus[k] = esf_full[k] - z_j * esf_minus[k-1]
/// Returns a vector of length m (one shorter than the full ESF).
[[nodiscard]] inline Eigen::VectorXd esf_excluding(
    const Eigen::VectorXd& esf_full,
    double z_j)
{
    const int m = static_cast<int>(esf_full.size()) - 1; // full vector had m elements
    assert(m >= 1);
    Eigen::VectorXd esf_minus = Eigen::VectorXd::Zero(m);
    esf_minus(0) = 1.0;

    for (int k = 1; k < m; ++k) {
        esf_minus(k) = esf_full(k) - z_j * esf_minus(k - 1);
    }

    return esf_minus;
}

/// Compute the falling factorial: P(n, k) = n! / (n-k)! = n * (n-1) * ... * (n-k+1).
/// Returns 1 if k == 0. Requires n >= k >= 0.
[[nodiscard]] inline double falling_factorial(int n, int k)
{
    assert(n >= k && k >= 0);
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result *= static_cast<double>(n - i);
    }
    return result;
}

} // namespace brew::multi_target
