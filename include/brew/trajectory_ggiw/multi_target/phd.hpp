#pragma once

// PHD RFS for the trajectory_ggiw package (concrete; brew::trajectory_ggiw).

#include "brew/trajectory_ggiw/trajectory_ggiw_model.hpp"

#include "brew/shared/rfs_base.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/shared/trajectory_window.hpp"
#include "brew/shared/filter_base.hpp"
#include "brew/shared/filter_traits.hpp"
#include "brew/trajectory_ggiw/filters/trajectory_ggiw_ekf.hpp"   // for the GGIW-measurement update
#include "brew/shared/fusion/prune.hpp"
#include "brew/trajectory_ggiw/merge.hpp"
#include "brew/shared/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <vector>

namespace brew::trajectory_ggiw {

template <typename Scalar = double, int D = Eigen::Dynamic, int De = Eigen::Dynamic, int MaxComponents = Eigen::Dynamic>
class PHD : public multi_target::RFSBase {
    using T = models::TrajectoryGGIW<Scalar, D, De>;
public:
    PHD() = default;

    [[nodiscard]] std::unique_ptr<multi_target::RFSBase> clone() const override {
        auto c = std::make_unique<PHD<Scalar, D, De, MaxComponents>>();
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
        c->detect_field_ = detect_field_;
        c->detect_radius_ = detect_radius_;
        c->prob_detection_out_ = prob_detection_out_;
        if (meas_mix_) c->meas_mix_ = meas_mix_->clone();
        return c;
    }

    // per-track detection probability from the observability field (see set_detection_field): uses the
    // trajectory's CURRENT (last) state position.
    [[nodiscard]] double pd_at(const T& traj) const {
        if (detect_radius_ <= 0.0 || detect_field_.cols() == 0) return prob_detection_;
        const Eigen::VectorXd mean = traj.current().mean();
        const Eigen::Index pd_dim = detect_field_.rows();
        const auto pos = mean.head(pd_dim);
        int n = 0;                                            // # satellite sub-points viewing the storm
        for (Eigen::Index c = 0; c < detect_field_.cols(); ++c)
            if ((detect_field_.col(c) - pos).norm() <= detect_radius_) ++n;
        // observability-GRADED detection probability: prob at least one of the n co-visible satellites
        // detects the storm this bin = 1-(1-p_sat)^n. n grows with coverage, so a well-observed storm
        // has Pd~1 (a miss really means gone) and a barely-observed one has a low Pd (a miss is expected)
        // -> the same filter behaves correctly at dense AND sparse constellations with no re-tuning.
        // n=0 (out of view) -> coast on prob_detection_out_ (~0), no missed-detection penalty.
        if (n == 0) return prob_detection_out_;
        return 1.0 - std::pow(1.0 - prob_detection_, n);
    }

    // observability field + GGIW measurement components (the fused PHD mixture, extent + rate carried)
    void set_detection_field(const Eigen::MatrixXd& field, double radius, double pd_out) {
        detect_field_ = field; detect_radius_ = radius; prob_detection_out_ = pd_out;
    }
    void set_measurement_components(const models::Mixture<models::GGIW<Scalar, D, De>>& z) {
        meas_mix_ = z.clone();
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
    void set_birth_weights(const Eigen::VectorXd& weights) {
        if (!birth_model_) return;
        if (weights.size() == 1) {
            birth_model_->weights().setConstant(weights(0));
        } else {
            birth_model_->weights() = weights;
        }
    }

    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::ClusterBase> obj) { cluster_obj_ = std::move(obj); }

    [[nodiscard]] const models::Mixture<T, MaxComponents>& intensity() const { return *intensity_; }
    [[nodiscard]] models::Mixture<T, MaxComponents>& intensity() { return *intensity_; }

    [[nodiscard]] const std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>>& extracted_mixtures() const {
        return extracted_mixtures_;
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
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !has_filter_) return;

        // GGIW-measurement mode: measurements supplied as GGIW components (set_measurement_components)
        // whose positions are the columns of ``measurements``; fuse via the GGIW measurement update.
        auto* gekf = dynamic_cast<filters::TrajectoryGGIWEKF<Scalar, D, De>*>(filter_.get());
        const bool use_ggiw = gekf && meas_mix_
                              && static_cast<int>(meas_mix_->size()) == measurements.cols();

        std::vector<Eigen::MatrixXd> meas_groups;
        if (!use_ggiw && is_extended_ && cluster_obj_) {
            meas_groups = cluster_obj_->cluster(measurements);
        } else {
            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }

        auto undetected = std::make_unique<models::Mixture<T, MaxComponents>>();
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            const auto& component = intensity_->component(i);
            const double pd = pd_at(component);
            const double weight = intensity_->weight(i);
            const double ordinary = weight * (1.0 - pd);
            if (ordinary > 0.0) {
                undetected->add_component(component.clone_typed(), ordinary);
            }
            const auto& current = component.current();
            const double q0 = std::pow(
                current.beta() / (current.beta() + 1.0), current.alpha());
            const double zero_count_weight = weight * pd * q0;
            if (zero_count_weight > 0.0) {
                auto zero_count = component.clone_typed();
                zero_count->current().beta() += 1.0;
                undetected->add_component(std::move(zero_count), zero_count_weight);
            }
        }
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            intensity_->weights()(static_cast<Eigen::Index>(i)) *= pd_at(intensity_->component(i));
        }

        models::Mixture<T, MaxComponents> corrected_mix;

        for (std::size_t z = 0; z < meas_groups.size(); ++z) {
            const Eigen::MatrixXd& meas = meas_groups[z];
            Eigen::VectorXd z_gate;
            if (is_extended_ && meas.cols() > 1) {
                z_gate = meas.rowwise().mean();
            } else {
                z_gate = meas.col(0);
            }

            Eigen::VectorXd meas_flat;
            if (meas.cols() > 1) {
                meas_flat.resize(meas.size());
                for (int j = 0; j < meas.cols(); ++j) {
                    meas_flat.segment(j * meas.rows(), meas.rows()) = meas.col(j);
                }
            } else {
                meas_flat = meas.col(0);
            }

            std::vector<double> w_lst;
            std::vector<std::unique_ptr<T>> d_lst;

            for (std::size_t k = 0; k < intensity_->size(); ++k) {
                double gate_val = use_ggiw
                    ? gekf->gate_component(meas_mix_->component(z), intensity_->component(k))
                    : filter_->gate(z_gate, intensity_->component(k));
                if (gate_val < gate_threshold_) {
                    auto [dist, qz] = use_ggiw
                        ? gekf->correct_component(meas_mix_->component(z), intensity_->component(k))
                        : filter_->correct(meas_flat, intensity_->component(k));
                    double w = qz * intensity_->weight(k);
                    w_lst.push_back(w);
                    d_lst.push_back(dist.clone_typed());
                }
            }

            double w_sum = 0.0;
            for (double w : w_lst) w_sum += w;
            const int W = static_cast<int>(meas.cols());
            double clutter_term = (W > 1)
                ? std::pow(clutter_rate_ * clutter_density_, W)
                : clutter_rate_ * clutter_density_;
            double denom = clutter_term + w_sum;
            if (!(denom > 0.0)) denom = 1.0;

            for (std::size_t i = 0; i < w_lst.size(); ++i) {
                double weight = w_lst[i] / denom;
                if (!std::isfinite(weight)) weight = 0.0;
                corrected_mix.add_component(std::move(d_lst[i]), weight);
            }
        }

        *intensity_ = std::move(*undetected);
        intensity_->add_components(corrected_mix);
        meas_mix_.reset();   // GGIW measurement components are consumed by this correct()
    }

    void cleanup() override {
        if (!intensity_) return;
        fusion::prune(*intensity_, prune_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));
        merge(*intensity_, merge_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));

        this->push_history(extracted_mixtures_, extract());
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
    std::unique_ptr<filters::Filter<T>> filter_;
    bool has_filter_ = false;
    std::unique_ptr<models::Mixture<T, MaxComponents>> intensity_;
    std::unique_ptr<models::Mixture<T, MaxComponents>> birth_model_;
    std::shared_ptr<clustering::ClusterBase> cluster_obj_;
    std::unique_ptr<models::Mixture<models::GGIW<Scalar, D, De>>> meas_mix_;   // GGIW measurements (transient)
    Eigen::MatrixXd detect_field_;          // observability field (state-dependent Pd)
    double detect_radius_ = -1.0;
    double prob_detection_out_ = 0.0;
    double prune_threshold_ = 1e-4;
    double merge_threshold_ = 4.0;
    int max_components_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    bool is_extended_ = false;
    std::deque<std::unique_ptr<models::Mixture<T, MaxComponents>>> extracted_mixtures_;
};

}
