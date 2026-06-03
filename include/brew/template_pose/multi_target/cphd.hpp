#pragma once

// CPHD RFS for the template_pose package (concrete; brew::template_pose).

#include "brew/template_pose/template_pose_model.hpp"

#include "brew/shared/rfs_base.hpp"
#include "brew/shared/elementary_symmetric.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/trajectory_window.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/filter_traits.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/template_pose/merge.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <type_traits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace brew::template_pose {

template <typename Scalar = double, int D = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class CPHD : public multi_target::RFSBase {
    using T = models::TemplatePose<Scalar, D>;
public:
    CPHD() = default;

    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto c = std::make_unique<CPHD<Scalar, D, MaxComponents>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        c->has_filter_ = has_filter_;
        if (intensity_) c->intensity_ = intensity_->clone();
        if (birth_model_) c->birth_model_ = birth_model_->clone();
        c->prune_threshold_ = prune_threshold_;
        c->merge_threshold_ = merge_threshold_;
        c->max_components_ = max_components_;
        c->extract_threshold_ = extract_threshold_;
        c->gate_threshold_ = gate_threshold_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        c->cardinality_ = cardinality_;
        c->birth_cardinality_ = birth_cardinality_;
        c->max_cardinality_ = max_cardinality_;
        return c;
    }

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
        has_filter_ = true;
    }

    void set_birth_model(std::unique_ptr<models::Mixture<T, MaxComponents>> birth) {
        if (birth && !birth->empty()) {
            is_extended_ = birth->component(0).is_extended();
            if (is_extended_ && !cluster_obj_) {
                cluster_obj_ = std::make_shared<clustering::DBSCAN>();
            }
        }
        birth_model_ = std::move(birth);
    }

    void set_intensity(std::unique_ptr<models::Mixture<T, MaxComponents>> intensity) {
        intensity_ = std::move(intensity);
    }

    void set_prune_threshold(double t) { prune_threshold_ = t; }
    void set_merge_threshold(double t) { merge_threshold_ = t; }
    void set_max_components(int n) { max_components_ = n; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::ClusterBase> obj) { cluster_obj_ = std::move(obj); }

    void set_cardinality(Eigen::VectorXd card) { cardinality_ = std::move(card); }

    void set_birth_cardinality(Eigen::VectorXd birth_card) { birth_cardinality_ = std::move(birth_card); }

    void set_max_cardinality(int n) { max_cardinality_ = n; }

    void set_poisson_cardinality(double lambda) {
        cardinality_ = make_poisson_pmf(lambda, max_cardinality_);
    }

    void set_poisson_birth_cardinality(double lambda) {
        birth_cardinality_ = make_poisson_pmf(lambda, max_cardinality_);
    }

    [[nodiscard]] const models::Mixture<T, MaxComponents>& intensity() const { return *intensity_; }
    [[nodiscard]] models::Mixture<T, MaxComponents>& intensity() { return *intensity_; }

    [[nodiscard]] const std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    [[nodiscard]] const Eigen::VectorXd& cardinality() const { return cardinality_; }

    [[nodiscard]] double estimated_cardinality() const {
        double mean = 0.0;
        for (int n = 0; n < cardinality_.size(); ++n) {
            mean += n * cardinality_(n);
        }
        return mean;
    }

    [[nodiscard]] int map_cardinality() const {
        int best = 0;
        for (int n = 1; n < cardinality_.size(); ++n) {
            if (cardinality_(n) > cardinality_(best)) best = n;
        }
        return best;
    }

    [[nodiscard]] const std::deque<Eigen::VectorXd>& cardinality_history() const {
        return cardinality_history_;
    }

    void predict(int , double dt) override {
        if (!intensity_ || !has_filter_) return;

        for (std::size_t k = 0; k < intensity_->size(); ++k) {
            intensity_->weights()(static_cast<Eigen::Index>(k)) *= prob_survive_;
        }
        filter_->predict_batch_dynamic(dt, *intensity_);

        if (birth_model_) {
            auto birth_copy = birth_model_->clone();
            intensity_->add_components(*birth_copy);

        }

        if (cardinality_.size() > 0 && birth_cardinality_.size() > 0) {

            Eigen::VectorXd surv = predict_cardinality_survival(prob_survive_);

            cardinality_ = convolve_cardinality(surv, birth_cardinality_);
        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !has_filter_) return;

        std::vector<Eigen::MatrixXd> meas_groups;
        if (is_extended_ && cluster_obj_) {
            meas_groups = cluster_obj_->cluster(measurements);
        } else {
            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }

        const int m = static_cast<int>(meas_groups.size());
        const int J = static_cast<int>(intensity_->size());
        const int N = static_cast<int>(cardinality_.size());
        const double kappa = clutter_rate_ * clutter_density_;
        const double q_d = 1.0 - prob_detection_;

        const Eigen::VectorXd card_pred = cardinality_;

        Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(J, m);
        std::vector<std::vector<std::unique_ptr<T>>> corrected_dists(J);
        for (int k = 0; k < J; ++k) corrected_dists[k].resize(m);

        for (int j = 0; j < m; ++j) {
            const Eigen::MatrixXd& meas = meas_groups[j];
            Eigen::VectorXd z_gate = (is_extended_ && meas.cols() > 1)
                ? Eigen::VectorXd(meas.rowwise().mean())
                : Eigen::VectorXd(meas.col(0));

            Eigen::VectorXd meas_flat;
            if (meas.cols() > 1) {
                meas_flat.resize(meas.size());
                for (int c = 0; c < meas.cols(); ++c)
                    meas_flat.segment(c * meas.rows(), meas.rows()) = meas.col(c);
            } else {
                meas_flat = meas.col(0);
            }

            for (int k = 0; k < J; ++k) {
                if (filter_->gate(z_gate, intensity_->component(k)) < gate_threshold_) {
                    auto [dist, qz] = filter_->correct(meas_flat, intensity_->component(k));
                    delta(k, j) = prob_detection_ * intensity_->weight(k) * qz;
                    corrected_dists[k][j] = dist.clone_typed();
                }
            }
        }

        Eigen::VectorXd Lambda(m);
        for (int j = 0; j < m; ++j) {
            double L_j = delta.col(j).sum();
            Lambda(j) = (kappa > 0.0) ? (L_j / kappa) : L_j;
        }
        Eigen::VectorXd esf_full = multi_target::elementary_symmetric_functions(Lambda);

        Eigen::VectorXd upsilon0 = compute_upsilon(esf_full, m, N, q_d, 0);

        double denom = card_pred.dot(upsilon0);
        if (denom <= 0.0) denom = 1e-300;

        Eigen::VectorXd card_updated(N);
        for (int n = 0; n < N; ++n)
            card_updated(n) = card_pred(n) * upsilon0(n);
        double card_sum = card_updated.sum();
        if (card_sum > 0.0) card_updated /= card_sum;
        cardinality_ = card_updated;

        Eigen::VectorXd upsilon1_full = compute_upsilon(esf_full, m, N, q_d, 1);
        double numer_undetected = card_pred.dot(upsilon1_full);
        double weight_ratio_undetected = numer_undetected / denom;

        auto updated_intensity = std::make_unique<models::Mixture<T, MaxComponents>>();

        for (int k = 0; k < J; ++k) {
            double w = q_d * intensity_->weight(k) * weight_ratio_undetected;
            updated_intensity->add_component(intensity_->component(k).clone_typed(), w);
        }

        for (int j = 0; j < m; ++j) {
            Eigen::VectorXd esf_minus_j = multi_target::esf_excluding(esf_full, Lambda(j));
            Eigen::VectorXd upsilon1_j = compute_upsilon(esf_minus_j, m - 1, N, q_d, 1);

            double numer_j = card_pred.dot(upsilon1_j);
            double weight_ratio_j = numer_j / denom;

            for (int k = 0; k < J; ++k) {
                if (delta(k, j) > 0.0 && corrected_dists[k][j]) {

                    double w = delta(k, j) * weight_ratio_j / kappa;
                    updated_intensity->add_component(
                        std::move(corrected_dists[k][j]), w);
                }
            }
        }

        intensity_ = std::move(updated_intensity);
    }

    void cleanup() override {
        if (!intensity_) return;
        fusion::prune(*intensity_, prune_threshold_);
        merge(*intensity_, merge_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));

        this->push_history(extracted_mixtures_, extract());
        this->push_history(cardinality_history_, cardinality_);
    }

    [[nodiscard]] std::unique_ptr<models::Mixture<T, MaxComponents>> extract() const {
        if (!intensity_) return nullptr;
        auto result = std::make_unique<models::Mixture<T, MaxComponents>>();
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            if (intensity_->weight(i) >= extract_threshold_) {
                result->add_component(
                    intensity_->component(i).clone_typed(),
                    intensity_->weight(i));
            }
        }
        return result;
    }

private:

    [[nodiscard]] static Eigen::VectorXd compute_upsilon(
        const Eigen::VectorXd& esf, int m_esf, int N, double q_d, int offset)
    {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(N);
        const double log_qd = (q_d > 0.0) ? std::log(q_d) : -700.0;

        for (int n = 0; n < N; ++n) {
            int j_max = std::min(m_esf, n - offset);
            if (j_max < 0) continue;

            double max_log = -std::numeric_limits<double>::infinity();
            std::vector<double> log_terms;
            log_terms.reserve(j_max + 1);

            for (int j = 0; j <= j_max; ++j) {
                if (j < esf.size() && n >= j + offset && esf(j) > 0.0) {
                    double log_esf = std::log(esf(j));

                    double log_ff = lgamma(n + 1) - lgamma(n - j - offset + 1);
                    double log_qd_term = (n - j - offset) * log_qd;
                    double lt = log_esf + log_ff + log_qd_term;
                    log_terms.push_back(lt);
                    max_log = std::max(max_log, lt);
                }
            }

            if (!log_terms.empty() && std::isfinite(max_log)) {
                double sum = 0.0;
                for (double lt : log_terms) {
                    sum += std::exp(lt - max_log);
                }
                result(n) = std::exp(max_log + std::log(sum));
            }
        }
        return result;
    }

    [[nodiscard]] static Eigen::VectorXd make_poisson_pmf(double lambda, int max_n) {
        Eigen::VectorXd pmf(max_n + 1);
        double log_lambda = std::log(lambda);
        double log_factorial = 0.0;
        for (int n = 0; n <= max_n; ++n) {
            pmf(n) = std::exp(n * log_lambda - lambda - log_factorial);
            log_factorial += std::log(static_cast<double>(n + 1));
        }
        pmf /= pmf.sum();
        return pmf;
    }

    [[nodiscard]] Eigen::VectorXd predict_cardinality_survival(double ps) const {
        const int N = static_cast<int>(cardinality_.size());
        Eigen::VectorXd surv = Eigen::VectorXd::Zero(N);

        for (int j = 0; j < N; ++j) {
            for (int i = j; i < N; ++i) {

                double log_binom = lgamma(i + 1) - lgamma(j + 1) - lgamma(i - j + 1);
                double val = std::exp(log_binom + j * std::log(ps)
                                      + (i - j) * std::log(1.0 - ps));
                surv(j) += val * cardinality_(i);
            }
        }

        return surv;
    }

    [[nodiscard]] Eigen::VectorXd convolve_cardinality(
        const Eigen::VectorXd& surv, const Eigen::VectorXd& birth) const
    {
        const int N = std::min(static_cast<int>(surv.size()),
                               max_cardinality_ + 1);
        Eigen::VectorXd result = Eigen::VectorXd::Zero(N);

        for (int n = 0; n < N; ++n) {
            for (int j = 0; j <= n; ++j) {
                if (j < surv.size() && (n - j) < birth.size()) {
                    result(n) += surv(j) * birth(n - j);
                }
            }
        }

        double s = result.sum();
        if (s > 0.0) result /= s;
        return result;
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    bool has_filter_ = false;
    std::unique_ptr<models::Mixture<T, MaxComponents>> intensity_;
    std::unique_ptr<models::Mixture<T, MaxComponents>> birth_model_;
    std::shared_ptr<clustering::ClusterBase> cluster_obj_;
    double prune_threshold_ = 1e-4;
    double merge_threshold_ = 4.0;
    int max_components_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    bool is_extended_ = false;

    Eigen::VectorXd cardinality_;
    Eigen::VectorXd birth_cardinality_;
    int max_cardinality_ = 100;
    std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>> extracted_mixtures_;
    std::deque<Eigen::VectorXd> cardinality_history_;
};

}
