#pragma once

#include <Eigen/Dense>
#include <cassert>

namespace brew::multi_target {

[[nodiscard]] inline Eigen::VectorXd elementary_symmetric_functions(
    const Eigen::VectorXd& z)
{
    const int m = static_cast<int>(z.size());
    Eigen::VectorXd esf = Eigen::VectorXd::Zero(m + 1);
    esf(0) = 1.0;

    for (int i = 0; i < m; ++i) {

        for (int j = i + 1; j >= 1; --j) {
            esf(j) += z(i) * esf(j - 1);
        }
    }

    return esf;
}

[[nodiscard]] inline Eigen::VectorXd esf_excluding(
    const Eigen::VectorXd& esf_full,
    double z_j)
{
    const int m = static_cast<int>(esf_full.size()) - 1;
    assert(m >= 1);
    Eigen::VectorXd esf_minus = Eigen::VectorXd::Zero(m);
    esf_minus(0) = 1.0;

    for (int k = 1; k < m; ++k) {
        esf_minus(k) = esf_full(k) - z_j * esf_minus(k - 1);
    }

    return esf_minus;
}

[[nodiscard]] inline double falling_factorial(int n, int k)
{
    assert(n >= k && k >= 0);
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result *= static_cast<double>(n - i);
    }
    return result;
}

}
