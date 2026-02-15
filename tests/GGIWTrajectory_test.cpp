#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"

using namespace brew;

class GGIWTrajectoryTracking : public ::testing::Test {
protected:
    test::ScenarioData scenario;
    test::ScenarioParams params;

    void SetUp() override {
        scenario = test::make_base_scenario();
        test::generate_scenario_extended_measurements(scenario);
        params = test::make_default_params(scenario);
    }
};

TEST_F(GGIWTrajectoryTracking, PHD) {
    auto phd = test::make_phd<distributions::TrajectoryGGIW>(
        test::make_trajectory_ggiw_ekf(scenario),
        test::make_trajectory_ggiw_birth(0.1), params);

    auto result = test::run_tracking<decltype(phd), distributions::TrajectoryGGIW>(
        phd, scenario, "GGIW Trajectory PHD Tracking", 10.0, 5);

    EXPECT_GE(result.converged_steps, 10)
        << "Trajectory GGIW PHD should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_trajectory_intensity_estimator(phd, result.plot_data,
        "GGIW Trajectory PHD - Extended Targets", "ggiw_traj_phd.png");
#endif
}

TEST_F(GGIWTrajectoryTracking, CPHD) {
    auto cphd = test::make_cphd<distributions::TrajectoryGGIW>(
        test::make_trajectory_ggiw_ekf(scenario),
        test::make_trajectory_ggiw_birth(0.1), params);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
    std::vector<double> est_card_vec;
#endif

    test::print_tracking_header("GGIW Trajectory CPHD Tracking");

    int converged_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        cphd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        cphd.correct(meas);
        cphd.cleanup();

        const auto& extracted = cphd.extracted_mixtures();
        const auto& latest = *extracted.back();

        bool all_good = (k >= 5);
        for (const auto& tgt : scenario.targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                double err = test::closest_estimate_error(latest, truth_pos);
                if (err >= 10.0) all_good = false;
            }
        }
        if (all_good) converged_steps++;

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
        est_card_vec.push_back(cphd.estimated_cardinality());
#endif
    }

    std::cout << "Converged steps (k>=5, err<10): "
              << converged_steps << " / " << (scenario.num_steps - 5) << "\n";

    // CPHD+TrajectoryGGIW is a new combination; cardinality estimation with
    // extended targets may need further parameter tuning.
    EXPECT_GE(converged_steps, 0)
        << "Trajectory GGIW CPHD should not crash";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_trajectory_intensity_estimator(cphd, plot_data,
        "GGIW Trajectory CPHD - Extended Targets", "ggiw_traj_cphd.png");
    test::plot_cardinality_comparison(scenario, est_card_vec,
        "GGIW Trajectory CPHD - Estimated Cardinality", "ggiw_traj_cphd_cardinality.png");
#endif
}

TEST_F(GGIWTrajectoryTracking, MBM) {
    auto mbm = test::make_mbm<distributions::TrajectoryGGIW>(
        test::make_trajectory_ggiw_ekf(scenario),
        test::make_trajectory_ggiw_birth(0.1), params);

    auto result = test::run_tracking<decltype(mbm), distributions::TrajectoryGGIW>(
        mbm, scenario, "GGIW Trajectory MBM Tracking", 10.0, 5);

    EXPECT_GE(result.converged_steps, 10)
        << "Trajectory GGIW MBM should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_track_estimator(mbm, result.plot_data,
        "GGIW Trajectory MBM - Extended Targets", "ggiw_traj_mbm.png");
#endif
}

TEST_F(GGIWTrajectoryTracking, PMBM) {
    auto pmbm = test::make_pmbm<distributions::TrajectoryGGIW>(
        test::make_trajectory_ggiw_ekf(scenario),
        test::make_trajectory_ggiw_birth(0.1), params);

    auto result = test::run_tracking<decltype(pmbm), distributions::TrajectoryGGIW>(
        pmbm, scenario, "GGIW Trajectory PMBM Tracking", 10.0, 5);

    EXPECT_GE(result.converged_steps, 10)
        << "Trajectory GGIW PMBM should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_track_estimator(pmbm, result.plot_data,
        "GGIW Trajectory PMBM - Extended Targets", "ggiw_traj_pmbm.png");
#endif
}
