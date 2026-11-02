#pragma once
#include "brew/utils/unit_sphere.hpp"
namespace brew::multiObjectTrackers {
// Angles in this configuration are degrees; state/filter covariances use radians.
// The time unit is chosen by the caller and must agree with the angular rates.
struct SphericalPHDConfig {
    double process_angle_var = 2e-7, process_rate_var = 2e-10;
    double measurement_sigma_deg = .25, temporal_decay = 1.01;
    double forgetting_factor = 7200., scaling_parameter = .3, ci_weight = .5;
    double prob_detection = .95, prob_survive = .997;
    double clutter_rate = 1e-5, clutter_density = 1.;
    double extract_threshold = .35, prune_threshold = 1e-5;
    double merge_threshold = 9., gate_threshold = 25.;
    int max_components = 32;
    double cluster_eps_deg = 4.;
    int cluster_min_pts = 3;
    double birth_weight = .25, birth_alpha = 12., birth_beta = 2.;
    double birth_angle_sigma_deg = 3., birth_rate_sigma_deg = .01;
    double birth_extent_sigma_deg = 2., birth_extent_dof = 10.;
    double birth_suppress_deg = 4., fov_radius_deg = 12., prob_detection_out = .02;

    Eigen::Matrix4d process_covariance() const {
        return Eigen::Vector4d(process_angle_var, process_angle_var,
                               process_rate_var, process_rate_var).asDiagonal();
    }

    template<class Filter, class PHDType>
    void configure(Filter& filter, PHDType& phd, double measurement_variance) const {
        filter.set_process_noise(process_covariance().template cast<typename Filter::DistScalar>());
        filter.set_measurement_noise((measurement_variance * Eigen::Matrix2d::Identity()).template cast<typename Filter::DistScalar>());
        filter.set_temporal_decay(temporal_decay);
        filter.set_forgetting_factor(forgetting_factor);
        filter.set_scaling_parameter(scaling_parameter);
        filter.set_ci_weight(ci_weight);
        phd.set_prob_detection(prob_detection);
        phd.set_prob_survive(prob_survive);
        phd.set_clutter_rate(clutter_rate);
        phd.set_clutter_density(clutter_density);
        phd.set_extract_threshold(extract_threshold);
        phd.set_prune_threshold(prune_threshold);
        phd.set_merge_threshold(merge_threshold);
        phd.set_gate_threshold(gate_threshold);
        phd.set_max_components(max_components);
    }
};

}
