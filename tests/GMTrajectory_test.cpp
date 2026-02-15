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

TEST_F(GMTrajectoryTracking, Comparison) {
    // PHD
    auto phd = test::make_phd<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.05), params);
    auto phd_result = test::run_tracking<decltype(phd), distributions::TrajectoryGaussian>(
        phd, scenario, "GM Trajectory PHD Tracking", 5.0, 10);
    EXPECT_GE(phd_result.converged_steps, 10)
        << "Trajectory Gaussian PHD should track both targets";

    // CPHD
    auto cphd = test::make_cphd<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.05), params);
    auto cphd_result = test::run_tracking_cphd<decltype(cphd), distributions::TrajectoryGaussian>(
        cphd, scenario, "GM Trajectory CPHD Tracking", 5.0, 10);
    EXPECT_GE(cphd_result.converged_steps, 10)
        << "Trajectory Gaussian CPHD should track both targets";

    // MBM
    auto mbm = test::make_mbm<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.1), params);
    auto mbm_result = test::run_tracking<decltype(mbm), distributions::TrajectoryGaussian>(
        mbm, scenario, "GM Trajectory MBM Tracking", 5.0, 10);
    EXPECT_GE(mbm_result.converged_steps, 10)
        << "Trajectory Gaussian MBM should track both targets";

    // PMBM
    auto pmbm = test::make_pmbm<distributions::TrajectoryGaussian>(
        test::make_trajectory_gaussian_ekf(scenario),
        test::make_trajectory_gaussian_birth(0.1), params);
    auto pmbm_result = test::run_tracking<decltype(pmbm), distributions::TrajectoryGaussian>(
        pmbm, scenario, "GM Trajectory PMBM Tracking", 5.0, 10);
    EXPECT_GE(pmbm_result.converged_steps, 10)
        << "Trajectory Gaussian PMBM should track both targets";

#ifdef BREW_ENABLE_PLOTTING
    // 2x2 comparison figure
    auto fig = test::create_comparison_figure();

    auto ax1 = test::comparison_subplot(fig, 0);
    test::populate_estimator_axes(ax1, phd, phd_result.plot_data, "PHD", true);

    auto ax2 = test::comparison_subplot(fig, 1);
    test::populate_estimator_axes(ax2, cphd, cphd_result.plot_data, "CPHD", true);

    auto ax3 = test::comparison_subplot(fig, 2);
    test::populate_estimator_axes(ax3, mbm, mbm_result.plot_data, "MBM", true);

    auto ax4 = test::comparison_subplot(fig, 3);
    test::populate_estimator_axes(ax4, pmbm, pmbm_result.plot_data, "PMBM", true);

    brew::plot_utils::save_figure(fig, test::output_dir() + "/gm_traj_comparison.png");

    // Cardinality figure
    test::plot_cardinality_comparison(scenario, cphd_result.cardinality,
        "GM Trajectory CPHD - Estimated Cardinality", "gm_traj_cphd_cardinality.png");
#endif
}
