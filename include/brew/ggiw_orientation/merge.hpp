#pragma once

// Component merge for the `ggiw_orientation` package -- owns ONLY the GGIWOrientation merge
// (no cross-model overload set). In namespace brew::ggiw_orientation so the package's
// concrete RFS resolve an unqualified merge(...) here.

#include "brew/shared/mixture.hpp"
#include "brew/ggiw_orientation/ggiw_orientation_model.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::ggiw_orientation {

template <typename Scalar, int D, int De, int N>
void merge(models::Mixture<models::GGIWOrientation<Scalar, D, De>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::GGIWOrientation<Scalar, D, De>, N> result;

    while (true) {

        double max_w = -1.0;
        std::size_t ref = 0;
        bool found = false;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (remaining[i] && mix.weight(i) > max_w) {
                max_w = mix.weight(i);
                ref = i;
                found = true;
            }
        }
        if (!found) break;

        Eigen::MatrixXd C_ref = mix.component(ref).covariance();
        C_ref = 0.5 * (C_ref + C_ref.transpose());
        Eigen::LLT<Eigen::MatrixXd> llt(C_ref);
        if (llt.info() != Eigen::Success) {
            const double eps_reg = 1e-9 * std::max(1.0, C_ref.trace() / C_ref.rows());
            C_ref += eps_reg * Eigen::MatrixXd::Identity(C_ref.rows(), C_ref.cols());
            llt.compute(C_ref);
        }

        std::vector<std::size_t> grp;
        for (std::size_t i = 0; i < mix.size(); ++i) {
            if (!remaining[i]) continue;
            const Eigen::VectorXd d = mix.component(i).mean() - mix.component(ref).mean();
            const Eigen::VectorXd y = llt.matrixL().solve(d);
            if (y.squaredNorm() <= threshold) {
                grp.push_back(i);
            }
        }

        double w_sum = 0.0;
        for (auto idx : grp) w_sum += mix.weight(idx);

        if (w_sum <= 0.0) {
            for (auto idx : grp) remaining[idx] = false;
            continue;
        }

        const int n = static_cast<int>(mix.component(ref).mean().size());
        const int d = mix.component(ref).extent_dim();

        Eigen::VectorXd m_new = Eigen::VectorXd::Zero(n);
        for (auto idx : grp) {
            const double q = mix.weight(idx) / w_sum;
            m_new += q * mix.component(idx).mean();
        }

        Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(n, n);
        double rate_mean = 0.0;
        double rate_second_moment = 0.0;
        double v_new = 0.0;
        Eigen::MatrixXd mean_extent = Eigen::MatrixXd::Zero(d, d);
        const double dof_floor = 2.0 * d + 2.0;

        for (auto idx : grp) {
            const double q = mix.weight(idx) / w_sum;
            const auto& component = mix.component(idx);
            const Eigen::VectorXd dm = component.mean() - m_new;
            P_new += q * (component.covariance() + dm * dm.transpose());
            const double component_rate_mean = component.alpha() / component.beta();
            rate_mean += q * component_rate_mean;
            rate_second_moment += q * component.alpha() * (component.alpha() + 1.0)
                                / (component.beta() * component.beta());
            v_new += q * component.v();
            mean_extent += q * component.V()
                         / std::max(component.v() - dof_floor, 1e-12);
        }
        P_new = 0.5 * (P_new + P_new.transpose());

        const double rate_variance = std::max(
            rate_second_moment - rate_mean * rate_mean, 1e-12);
        const double a_new = rate_mean * rate_mean / rate_variance;
        const double b_new = rate_mean / rate_variance;
        Eigen::MatrixXd V_new = std::max(v_new - dof_floor, 1e-12) * mean_extent;
        V_new = 0.5 * (V_new + V_new.transpose());

        result.add_component(
            std::make_unique<models::GGIWOrientation<Scalar, D, De>>(a_new, b_new, m_new, P_new, v_new, V_new),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

}
