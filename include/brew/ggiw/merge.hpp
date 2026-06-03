#pragma once

// Component merge for the `ggiw` package -- owns ONLY the GGIW merge
// (no cross-model overload set). In namespace brew::ggiw so the package's
// concrete RFS resolve an unqualified merge(...) here.

#include "brew/shared/mixture.hpp"
#include "brew/ggiw/ggiw_model.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace brew::ggiw {

template <typename Scalar, int D, int De, int N>
void merge(models::Mixture<models::GGIW<Scalar, D, De>, N>& mix, double threshold) {
    if (mix.size() < 2) return;

    std::vector<bool> remaining(mix.size(), true);
    models::Mixture<models::GGIW<Scalar, D, De>, N> result;

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
        Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(n, n);
        double a_new = 0.0, b_new = 0.0, v_new = 0.0;
        Eigen::MatrixXd V_new = Eigen::MatrixXd::Zero(d, d);

        for (auto idx : grp) {
            const double w = mix.weight(idx);
            m_new += w * mix.component(idx).mean();
            P_new += w * mix.component(idx).covariance();
            a_new += w * mix.component(idx).alpha();
            b_new += w * mix.component(idx).beta();
            v_new += w * mix.component(idx).v();
            V_new += w * mix.component(idx).V();
        }
        m_new /= w_sum;
        P_new /= w_sum;
        P_new = 0.5 * (P_new + P_new.transpose());
        a_new /= w_sum;
        b_new /= w_sum;
        v_new /= w_sum;
        V_new /= w_sum;
        V_new = 0.5 * (V_new + V_new.transpose());

        result.add_component(
            std::make_unique<models::GGIW<Scalar, D, De>>(a_new, b_new, m_new, P_new, v_new, V_new),
            w_sum);

        for (auto idx : grp) remaining[idx] = false;
    }

    mix = std::move(result);
}

}
