#pragma once

#include "brew/multi_target/rfs_base.hpp"
#include "brew/distributions/mixture.hpp"
#include "brew/distributions/trajectory_base_model.hpp"
#include "brew/filters/filter.hpp"
#include "brew/fusion/prune.hpp"
#include "brew/fusion/merge.hpp"
#include "brew/fusion/cap.hpp"
#include "brew/clustering/dbscan.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace brew::multi_target {

/// Probability Hypothesis Density (PHD) filter.
/// Template parameter T is the single distribution type (e.g., Gaussian, GGIW).
template <typename T>
class PHD : public RFSBase {
public:
    PHD() = default;

    [[nodiscard]] std::unique_ptr<RFSBase> clone() const override {
        auto c = std::make_unique<PHD<T>>();
        c->prob_detection_ = prob_detection_;
        c->prob_survive_ = prob_survive_;
        c->clutter_rate_ = clutter_rate_;
        c->clutter_density_ = clutter_density_;
        if (filter_) c->filter_ = filter_->clone();
        if (intensity_) c->intensity_ = intensity_->clone();
        if (birth_model_) c->birth_model_ = birth_model_->clone();
        c->prune_threshold_ = prune_threshold_;
        c->merge_threshold_ = merge_threshold_;
        c->max_components_ = max_components_;
        c->extract_threshold_ = extract_threshold_;
        c->gate_threshold_ = gate_threshold_;
        c->is_extended_ = is_extended_;
        if (cluster_obj_) c->cluster_obj_ = cluster_obj_;
        return c;
    }

    // ---- Configuration ----

    void set_filter(std::unique_ptr<filters::Filter<T>> filter) {
        filter_ = std::move(filter);
    }

    void set_birth_model(std::unique_ptr<distributions::Mixture<T>> birth) {
        birth_model_ = std::move(birth);
    }

    void set_intensity(std::unique_ptr<distributions::Mixture<T>> intensity) {
        intensity_ = std::move(intensity);
    }

    void set_prune_threshold(double t) { prune_threshold_ = t; }
    void set_merge_threshold(double t) { merge_threshold_ = t; }
    void set_max_components(int n) { max_components_ = n; }
    void set_extract_threshold(double t) { extract_threshold_ = t; }
    void set_gate_threshold(double t) { gate_threshold_ = t; }
    void set_extended_target(bool ext) { is_extended_ = ext; }
    void set_cluster_object(std::shared_ptr<clustering::DBSCAN> obj) { cluster_obj_ = std::move(obj); }

    [[nodiscard]] const distributions::Mixture<T>& intensity() const { return *intensity_; }
    [[nodiscard]] distributions::Mixture<T>& intensity() { return *intensity_; }

    [[nodiscard]] const std::vector<std::unique_ptr<distributions::Mixture<T>>>& extracted_mixtures() const {
        return extracted_mixtures_;
    }

    // ---- RFS interface ----

    void predict(int /*timestep*/, double dt) override {
        if (!intensity_ || !filter_) return;

        // Propagate surviving components and scale weights
        for (std::size_t k = 0; k < intensity_->size(); ++k) {
            intensity_->weights()(static_cast<Eigen::Index>(k)) *= prob_survive_;
            auto predicted = filter_->predict(dt, intensity_->component(k));
            *intensity_->components()[k] = std::move(predicted);
        }

        // Add birth components
        if (birth_model_) {
            auto birth_copy = birth_model_->clone();
            intensity_->add_components(*birth_copy);

            // Increment trajectory init_idx for birth components
            for (std::size_t k = 0; k < birth_model_->size(); ++k) {
                increment_init_idx(birth_model_->component(k));
            }
        }
    }

    void correct(const Eigen::MatrixXd& measurements) override {
        if (!intensity_ || !filter_) return;

        // Build measurement groups (cells)
        std::vector<Eigen::MatrixXd> meas_groups;
        if (is_extended_ && cluster_obj_) {
            meas_groups = cluster_obj_->cluster(measurements);
        } else {
            // Each column is a separate measurement
            for (int j = 0; j < measurements.cols(); ++j) {
                meas_groups.push_back(measurements.col(j));
            }
        }

        // Undetected copy: (1 - pd) * w
        auto undetected = intensity_->clone();
        for (std::size_t i = 0; i < undetected->size(); ++i) {
            undetected->weights()(static_cast<Eigen::Index>(i)) *= (1.0 - prob_detection_);
        }

        // Scale intensity weights by pd for correction
        for (std::size_t i = 0; i < intensity_->size(); ++i) {
            intensity_->weights()(static_cast<Eigen::Index>(i)) *= prob_detection_;
        }

        // Per-measurement correction
        distributions::Mixture<T> corrected_mix;

        for (std::size_t z = 0; z < meas_groups.size(); ++z) {
            const Eigen::MatrixXd& meas = meas_groups[z];
            Eigen::VectorXd z_gate;
            if (is_extended_ && meas.cols() > 1) {
                z_gate = meas.rowwise().mean();
            } else {
                z_gate = meas.col(0);
            }

            // Flatten measurement for correction (stack columns)
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
                double gate_val = filter_->gate(z_gate, intensity_->component(k));
                if (gate_val < gate_threshold_) {
                    auto [dist, qz] = filter_->correct(meas_flat, intensity_->component(k));
                    double w = qz * intensity_->weight(k);
                    w_lst.push_back(w);
                    d_lst.push_back(dist.clone_typed());
                }
            }

            // Normalize weights with clutter
            double w_sum = 0.0;
            for (double w : w_lst) w_sum += w;
            double denom = clutter_rate_ * clutter_density_ + w_sum;

            for (std::size_t i = 0; i < w_lst.size(); ++i) {
                corrected_mix.add_component(std::move(d_lst[i]), w_lst[i] / denom);
            }
        }

        // Replace intensity with undetected + corrected
        *intensity_ = std::move(*undetected);
        intensity_->add_components(corrected_mix);
    }

    void cleanup() override {
        if (!intensity_) return;
        fusion::prune(*intensity_, prune_threshold_);
        fusion::merge(*intensity_, merge_threshold_);
        fusion::cap(*intensity_, static_cast<std::size_t>(max_components_));

        // Extract and store
        extracted_mixtures_.push_back(extract());
    }

    /// Extract state estimates (components with weight >= threshold).
    [[nodiscard]] std::unique_ptr<distributions::Mixture<T>> extract() const {
        if (!intensity_) return nullptr;
        auto result = std::make_unique<distributions::Mixture<T>>();
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
    // Helper to increment init_idx for trajectory types (no-op for non-trajectory)
    template <typename U>
    static void increment_init_idx(U& /*dist*/) {
        // Default no-op for non-trajectory types
    }

    static void increment_init_idx(distributions::TrajectoryBaseModel& dist) {
        dist.init_idx += 1;
    }

    std::unique_ptr<filters::Filter<T>> filter_;
    std::unique_ptr<distributions::Mixture<T>> intensity_;
    std::unique_ptr<distributions::Mixture<T>> birth_model_;
    std::shared_ptr<clustering::DBSCAN> cluster_obj_;
    double prune_threshold_ = 1e-4;
    double merge_threshold_ = 4.0;
    int max_components_ = 100;
    double extract_threshold_ = 0.5;
    double gate_threshold_ = 9.0;
    bool is_extended_ = false;
    std::vector<std::unique_ptr<distributions::Mixture<T>>> extracted_mixtures_;
};

} // namespace brew::multi_target
