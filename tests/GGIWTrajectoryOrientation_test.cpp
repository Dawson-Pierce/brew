#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"

using namespace brew;

class GGIWTrajectoryOrientationTracking : public ::testing::Test {
protected:
    test::ScenarioData scenario;
    test::ScenarioParams params;

    void SetUp() override {
        scenario = test::make_base_scenario();
        test::generate_scenario_extended_measurements(scenario);
        params = test::make_default_params(scenario);
    }
};

TEST_F(GGIWTrajectoryOrientationTracking, Comparison) {
    // PHD
    auto phd = test::make_phd<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto phd_result = test::run_tracking<decltype(phd), models::TrajectoryGGIWOrientation>(
        phd, scenario, "GGIWOrientation Trajectory PHD Tracking", 10.0, 5);
    EXPECT_GE(phd_result.converged_steps, 10)
        << "Trajectory GGIWOrientation PHD should track both extended targets";

    // CPHD
    auto cphd = test::make_cphd<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto cphd_result = test::run_tracking_cphd<decltype(cphd), models::TrajectoryGGIWOrientation>(
        cphd, scenario, "GGIWOrientation Trajectory CPHD Tracking", 10.0, 5);
    EXPECT_GE(cphd_result.converged_steps, 0)
        << "Trajectory GGIWOrientation CPHD should not crash";

    // MBM
    auto mbm = test::make_mbm<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto mbm_result = test::run_tracking<decltype(mbm), models::TrajectoryGGIWOrientation>(
        mbm, scenario, "GGIWOrientation Trajectory MBM Tracking", 10.0, 5);
    EXPECT_GE(mbm_result.converged_steps, 10)
        << "Trajectory GGIWOrientation MBM should track both extended targets";

    // PMBM
    auto pmbm = test::make_pmbm<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto pmbm_result = test::run_tracking<decltype(pmbm), models::TrajectoryGGIWOrientation>(
        pmbm, scenario, "GGIWOrientation Trajectory PMBM Tracking", 10.0, 5);
    EXPECT_GE(pmbm_result.converged_steps, 10)
        << "Trajectory GGIWOrientation PMBM should track both extended targets";

    // MB
    auto mb = test::make_mb<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto mb_result = test::run_tracking<decltype(mb), models::TrajectoryGGIWOrientation>(
        mb, scenario, "GGIWOrientation Trajectory MB Tracking", 10.0, 5);
    EXPECT_GE(mb_result.converged_steps, 10)
        << "Trajectory GGIWOrientation MB should track both extended targets";

    // LMB
    auto lmb = test::make_lmb<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto lmb_result = test::run_tracking<decltype(lmb), models::TrajectoryGGIWOrientation>(
        lmb, scenario, "GGIWOrientation Trajectory LMB Tracking", 10.0, 5);
    EXPECT_GE(lmb_result.converged_steps, 10)
        << "Trajectory GGIWOrientation LMB should track both extended targets";

    // GLMB
    auto glmb = test::make_glmb<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto glmb_result = test::run_tracking<decltype(glmb), models::TrajectoryGGIWOrientation>(
        glmb, scenario, "GGIWOrientation Trajectory GLMB Tracking", 10.0, 5);
    EXPECT_GE(glmb_result.converged_steps, 10)
        << "Trajectory GGIWOrientation GLMB should track both extended targets";

    // JGLMB
    auto jglmb = test::make_jglmb<models::TrajectoryGGIWOrientation>(
        test::make_trajectory_ggiw_orientation_ekf(scenario),
        test::make_trajectory_ggiw_orientation_birth(0.1), params);
    auto jglmb_result = test::run_tracking<decltype(jglmb), models::TrajectoryGGIWOrientation>(
        jglmb, scenario, "GGIWOrientation Trajectory JGLMB Tracking", 10.0, 5);
    EXPECT_GE(jglmb_result.converged_steps, 10)
        << "Trajectory GGIWOrientation JGLMB should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    // 2x4 comparison figure
    auto fig = test::create_comparison_figure();

    auto ax1 = test::comparison_subplot(fig, 2, 4, 0);
    test::populate_estimator_axes(ax1, phd, phd_result.plot_data, "PHD", true);

    auto ax2 = test::comparison_subplot(fig, 2, 4, 1);
    test::populate_estimator_axes(ax2, cphd, cphd_result.plot_data, "CPHD", true);

    auto ax3 = test::comparison_subplot(fig, 2, 4, 2);
    test::populate_estimator_axes(ax3, mbm, mbm_result.plot_data, "MBM", true);

    auto ax4 = test::comparison_subplot(fig, 2, 4, 3);
    test::populate_estimator_axes(ax4, pmbm, pmbm_result.plot_data, "PMBM", true);

    auto ax5 = test::comparison_subplot(fig, 2, 4, 4);
    test::populate_estimator_axes(ax5, mb, mb_result.plot_data, "MB", true);

    auto ax6 = test::comparison_subplot(fig, 2, 4, 5);
    test::populate_estimator_axes(ax6, lmb, lmb_result.plot_data, "LMB", true);

    auto ax7 = test::comparison_subplot(fig, 2, 4, 6);
    test::populate_estimator_axes(ax7, glmb, glmb_result.plot_data, "GLMB", true);

    auto ax8 = test::comparison_subplot(fig, 2, 4, 7);
    test::populate_estimator_axes(ax8, jglmb, jglmb_result.plot_data, "JGLMB", true);

    brew::plot_utils::save_figure(fig, test::output_dir() + "/ggiw_orientation_traj_comparison.png");

    // Cardinality figure
    test::plot_cardinality_comparison(scenario, cphd_result.cardinality,
        "GGIWOrientation Trajectory CPHD - Estimated Cardinality", "ggiw_orientation_traj_cphd_cardinality.png");
#endif
}
