#pragma once

#include "brew/utils/fusion/gci_detail.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/usggiw.hpp"
#include <limits>
#include <numeric>
#include <vector>

namespace brew::utils::fusion {
using namespace brew::utils::sphere;

template<typename Scalar = double, int De = Eigen::Dynamic, int N = Eigen::Dynamic>
models::Mixture<models::USGGIW<Scalar, De>, N> gci(
    const std::vector<const models::Mixture<models::USGGIW<Scalar, De>, N>*>& sources,
    std::vector<double> weights = {}, double prune_threshold = 0.0) {
    using Model = models::USGGIW<Scalar, De>;
    namespace fd = utils::fusion::detail;
    models::Mixture<Model, N> out;
    std::vector<const models::Mixture<Model, N>*> src;
    std::vector<double> omega;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (sources[i] && !sources[i]->empty()) {
            src.push_back(sources[i]);
            omega.push_back(weights.size() == sources.size() ? weights[i] : 1.0);
        }
    }
    if (src.empty()) return out;
    fd::normalize_weights(omega);
    std::vector<std::size_t> index(src.size(), 0), sizes(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) sizes[i] = src[i]->size();
    struct Fused {
        double alpha, beta, dof, log_weight;
        Eigen::Vector3d direction;
        Eigen::Vector2d velocity;
        Eigen::Matrix4d covariance;
        Eigen::MatrixXd V;
    };
    std::vector<Fused> fused;
    do {
        Eigen::Vector3d reference_resultant = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < src.size(); ++i)
            reference_resultant += omega[i]
                * src[i]->component(index[i]).direction().template cast<double>();
        const Eigen::Vector3d reference = reference_resultant.norm() > 1e-12
            ? reference_resultant.normalized()
            : normalized(src[0]->component(index[0]).direction().template cast<double>());
        Eigen::Matrix4d precision = Eigen::Matrix4d::Zero();
        Eigen::Vector4d information = Eigen::Vector4d::Zero();
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(
            src[0]->component(index[0]).extent_dim(), src[0]->component(index[0]).extent_dim());
        double alpha = 0.0, beta = 0.0, dof = 0.0, logw = 0.0;
        double quad_sum = 0.0, logdet_sum = 0.0;
        double gamma_acc = 0.0, iw_acc = 0.0;
        const int de = src[0]->component(index[0]).extent_dim();
        for (std::size_t i = 0; i < src.size(); ++i) {
            const auto& model = src[i]->component(index[i]);
            const Eigen::Vector3d model_direction = model.direction().template cast<double>();
            const Eigen::Matrix2d T = transport_matrix(model_direction, reference);
            const Eigen::Matrix4d G = state_transport_matrix(model_direction, reference);
            Eigen::Vector4d x;
            x.head<2>() = local_error(reference, model_direction);
            x.tail<2>() = T * model.angular_velocity().template cast<double>();
            const Eigen::Matrix4d covariance =
                G * model.covariance().template cast<double>() * G.transpose();
            const Eigen::Matrix4d Pinv = covariance.ldlt().solve(Eigen::Matrix4d::Identity());
            precision += omega[i] * Pinv;
            information += omega[i] * Pinv * x;
            quad_sum += omega[i] * x.dot(Pinv * x);
            logdet_sum += omega[i] * fd::log_det_chol(covariance);
            alpha += omega[i] * model.alpha();
            beta += omega[i] * model.beta();
            dof += omega[i] * model.v();
            const Eigen::MatrixXd transported_V =
                T * model.V().template cast<double>() * T.transpose();
            V += omega[i] * transported_V;
            gamma_acc += omega[i] * (model.alpha() * std::log(model.beta())
                                      - std::lgamma(model.alpha()));
            const double ai = 0.5 * (model.v() - de - 1.0);
            iw_acc += omega[i] * (ai * fd::log_det_chol(transported_V)
                                   - fd::log_mv_gamma(ai, de));
            logw += omega[i] * std::log(std::max(src[i]->weight(index[i]), 1e-300));
        }
        const Eigen::Matrix4d P = precision.ldlt().solve(Eigen::Matrix4d::Identity());
        const Eigen::Vector4d x = P * information;
        const double log_c_gaussian = 0.5 * fd::log_det_chol(P) - 0.5 * logdet_sum
            - 0.5 * (quad_sum - x.dot(precision * x));
        const double log_c_gamma = gamma_acc + std::lgamma(alpha)
            - alpha * std::log(beta);
        const double af_iw = 0.5 * (dof - de - 1.0);
        const double log_c_iw = iw_acc - af_iw * fd::log_det_chol(V)
            + fd::log_mv_gamma(af_iw, de);
        const Eigen::Vector3d direction = apply_position_error(reference, x.head<2>());
        const Eigen::Matrix2d T = transport_matrix(reference, direction);
        const Eigen::Matrix4d G = state_transport_matrix(reference, direction);
        fused.push_back({alpha, beta, dof,
                         logw + log_c_gaussian + log_c_gamma + log_c_iw, direction,
                         T * x.tail<2>(), G * P * G.transpose(), T * V * T.transpose()});
        std::size_t j = src.size();
        while (j-- > 0) {
            if (++index[j] < sizes[j]) break;
            index[j] = 0;
        }
        if (j == static_cast<std::size_t>(-1)) break;
    } while (true);
    double maxlog = -std::numeric_limits<double>::infinity();
    for (const auto& f : fused) maxlog = std::max(maxlog, f.log_weight);
    double norm = 0.0;
    for (const auto& f : fused) norm += std::exp(f.log_weight - maxlog);
    for (const auto& f : fused) {
        const double w = std::exp(f.log_weight - maxlog) / norm;
        if (w < prune_threshold) continue;
        out.add_component(std::make_unique<Model>(
            f.alpha, f.beta, f.direction.template cast<Scalar>(), f.velocity.template cast<Scalar>(),
            f.covariance.template cast<Scalar>(), f.dof, f.V.template cast<Scalar>()), w);
    }
    return out;
}

using SphericalMixture = models::Mixture<models::USGGIW<>>;

// Stable proximity groups, with the strongest report first. Each group is fused
// once using equal-weight N-source GCI; its existence weight is the largest input.
class ReportFusion {
public:
    explicit ReportFusion(double gate_deg = 5.) : gate_deg_(gate_deg) {}

    static std::unique_ptr<SphericalMixture> unite(const std::vector<const SphericalMixture*>& inputs) {
        auto result = std::make_unique<SphericalMixture>();
        for (const auto* input : inputs) result->add_components(*input);
        return result;
    }

    std::unique_ptr<SphericalMixture> reduce(const SphericalMixture& input) const {
        auto result = std::make_unique<SphericalMixture>();
        std::vector<std::size_t> remaining(input.size());
        std::iota(remaining.begin(), remaining.end(), 0);
        while (!remaining.empty()) {
            const auto reference = *std::max_element(remaining.begin(), remaining.end(),
                [&](auto a, auto b) { return input.weight(a) < input.weight(b); });
            std::vector<std::size_t> group{reference};
            for (auto i : remaining)
                if (i != reference && std::acos(std::clamp(input.component(i).direction().dot(
                        input.component(reference).direction()), -1., 1.)) * (180./M_PI) <= gate_deg_)
                    group.push_back(i);
            if (group.size() == 1) result->add_component(input.component(reference), input.weight(reference));
            else {
                std::vector<SphericalMixture> sources(group.size());
                std::vector<const SphericalMixture*> pointers;
                for (std::size_t j = 0; j < group.size(); ++j) {
                    sources[j].add_component(input.component(group[j]), 1.);
                    pointers.push_back(&sources[j]);
                }
                auto fused = gci(pointers, std::vector<double>(group.size(), 1./group.size()), 0.);
                result->add_component(fused.component(0), input.weight(reference));
            }
            std::erase_if(remaining, [&](auto i) { return std::find(group.begin(), group.end(), i) != group.end(); });
        }
        return result;
    }

    struct Result {
        std::vector<std::unique_ptr<SphericalMixture>> rings;
        std::unique_ptr<SphericalMixture> pooled;
    };

    Result fuse_rings(const std::vector<std::vector<const SphericalMixture*>>& ordered_reports) const {
        Result result;
        std::vector<const SphericalMixture*> rings;
        for (const auto& reports : ordered_reports) {
            result.rings.push_back(reduce(*unite(reports)));
            rings.push_back(result.rings.back().get());
        }
        result.pooled = reduce(*unite(rings));
        return result;
    }

private:
    double gate_deg_;
};

}  // namespace brew::utils::fusion
