#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"

using namespace brew;

class GMTrajectoryTracking : public ::testing::Test {
protected:
    test::ScenarioData scenario;
    test::ScenarioParams params;

    void SetUp() override {
        scenario = test::make_base_scenario();
        test::generate_scenario_point_measurements(scenario);
        params = test::make_default_params(scenario);
    }
};

TEST_F(GMTrajectoryTracking, PHD) {
    auto phd = test::make_phd<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.05), params);

    auto result = test::run_tracking<decltype(phd), distributions::TrajectoryGaussian>(
        phd, scenario, "GM Trajectory PHD Tracking", 5.0, 10);

    EXPECT_GE(result.converged_steps, 10)
        << "Trajectory Gaussian PHD should track both targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_trajectory_intensity_estimator(phd, result.plot_data,
        "GM Trajectory PHD - Two Point Targets", "gm_traj_phd.png");
#endif
}

TEST_F(GMTrajectoryTracking, CPHD) {
    auto cphd = test::make_cphd<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.05), params);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
    std::vector<double> est_card_vec;
#endif

    test::print_tracking_header("GM Trajectory CPHD Tracking");

    int converged_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        cphd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        cphd.correct(meas);
        cphd.cleanup();

        const auto& extracted = cphd.extracted_mixtures();
        const auto& latest = *extracted.back();

        bool all_good = (k >= 10);
        for (const auto& tgt : scenario.targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                double err = test::closest_estimate_error(latest, truth_pos);
                if (err >= 5.0) all_good = false;
            }
        }
        if (all_good) converged_steps++;

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
        est_card_vec.push_back(cphd.estimated_cardinality());
#endif
    }

    std::cout << "Converged steps (k>=10, err<5): "
              << converged_steps << " / " << (scenario.num_steps - 10) << "\n";

    EXPECT_GE(converged_steps, 10)
        << "Trajectory Gaussian CPHD should track both targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_trajectory_intensity_estimator(cphd, plot_data,
        "GM Trajectory CPHD - Two Point Targets", "gm_traj_cphd.png");
    test::plot_cardinality_comparison(scenario, est_card_vec,
        "GM Trajectory CPHD - Estimated Cardinality", "gm_traj_cphd_cardinality.png");
#endif
}

TEST_F(GMTrajectoryTracking, MBM) {
    auto mbm = test::make_mbm<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.1), params);

    auto result = test::run_tracking<decltype(mbm), distributions::TrajectoryGaussian>(
        mbm, scenario, "GM Trajectory MBM Tracking", 5.0, 10);

    EXPECT_GE(result.converged_steps, 10)
        << "Trajectory Gaussian MBM should track both targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_track_estimator(mbm, result.plot_data,
        "GM Trajectory MBM - Two Point Targets", "gm_traj_mbm.png");
#endif
}

TEST_F(GMTrajectoryTracking, PMBM) {
    auto pmbm = test::make_pmbm<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.1), params);

    auto result = test::run_tracking<decltype(pmbm), distributions::TrajectoryGaussian>(
        pmbm, scenario, "GM Trajectory PMBM Tracking", 5.0, 10);

    EXPECT_GE(result.converged_steps, 10)
        << "Trajectory Gaussian PMBM should track both targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_track_estimator(pmbm, result.plot_data,
        "GM Trajectory PMBM - Two Point Targets", "gm_traj_pmbm.png");
#endif
}
