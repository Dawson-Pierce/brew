#pragma once

#include "brew/assignment/hungarian.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace brew::metrics {

struct OSPAResult {
    double distance = 0.0;
    double localization = 0.0;
    double cardinality = 0.0;
};

[[nodiscard]] inline OSPAResult calculate_ospa(
    const std::vector<Eigen::VectorXd>& estimates,
    const std::vector<Eigen::VectorXd>& truths,
    double cutoff, int power)
{
    OSPAResult result;
    const int m = static_cast<int>(estimates.size());
    const int n = static_cast<int>(truths.size());

    if (m == 0 && n == 0) return result;

    if (m == 0 || n == 0) {
        result.distance = cutoff;
        result.localization = 0.0;
        result.cardinality = cutoff;
        return result;
    }

    const double p = static_cast<double>(power);
    const int max_mn = std::max(m, n);

    Eigen::MatrixXd cost(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double d = (estimates[i] - truths[j]).norm();
            cost(i, j) = std::min(d, cutoff);
        }
    }

    Eigen::MatrixXd cost_p(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cost_p(i, j) = std::pow(cost(i, j), p);
        }
    }

    auto assignment = assignment::hungarian(cost_p);

    double loc_sum = 0.0;
    for (const auto& [row, col] : assignment.assignments) {
        loc_sum += std::pow(cost(row, col), p);
    }

    double card_sum = std::abs(m - n) * std::pow(cutoff, p);

    result.localization = std::pow(loc_sum / max_mn, 1.0 / p);
    result.cardinality = std::pow(card_sum / max_mn, 1.0 / p);
    result.distance = std::pow((loc_sum + card_sum) / max_mn, 1.0 / p);

    return result;
}

[[nodiscard]] inline OSPAResult calculate_gospa(
    const std::vector<Eigen::VectorXd>& estimates,
    const std::vector<Eigen::VectorXd>& truths,
    double cutoff, int power, double alpha = 2.0)
{
    OSPAResult result;
    const int m = static_cast<int>(estimates.size());
    const int n = static_cast<int>(truths.size());

    if (m == 0 && n == 0) return result;

    const double p = static_cast<double>(power);

    if (m == 0 || n == 0) {
        int num_unassigned = std::max(m, n);
        double card_sum = num_unassigned * std::pow(cutoff, p) / alpha;
        result.cardinality = std::pow(card_sum, 1.0 / p);
        result.distance = result.cardinality;
        return result;
    }

    Eigen::MatrixXd cost(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double d = (estimates[i] - truths[j]).norm();
            cost(i, j) = std::min(d, cutoff);
        }
    }

    Eigen::MatrixXd cost_p(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cost_p(i, j) = std::pow(cost(i, j), p);
        }
    }

    auto assignment = assignment::hungarian(cost_p);

    double loc_sum = 0.0;
    for (const auto& [row, col] : assignment.assignments) {
        loc_sum += std::pow(cost(row, col), p);
    }

    int n_assigned = static_cast<int>(assignment.assignments.size());
    int n_unassigned = (m - n_assigned) + (n - n_assigned);
    double card_sum = n_unassigned * std::pow(cutoff, p) / alpha;

    result.localization = std::pow(loc_sum, 1.0 / p);
    result.cardinality = std::pow(card_sum, 1.0 / p);
    result.distance = std::pow(loc_sum + card_sum, 1.0 / p);

    return result;
}

}
