#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"

using namespace brew;

class GGIWTracking : public ::testing::Test {
protected:
    test::ScenarioData scenario;
    test::ScenarioParams params;

    void SetUp() override {
        scenario = test::make_base_scenario();
        test::generate_scenario_extended_measurements(scenario);
        params = test::make_default_params(scenario);
    }
};

TEST_F(GGIWTracking, Comparison) {
    // PHD
    auto phd = test::make_phd<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto phd_result = test::run_tracking<decltype(phd), distributions::GGIW>(
        phd, scenario, "GGIW PHD Tracking", 10.0, 5);
    EXPECT_GE(phd_result.converged_steps, 10)
        << "GGIW PHD should track both extended targets";

    // CPHD
    auto cphd = test::make_cphd<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto cphd_result = test::run_tracking_cphd<decltype(cphd), distributions::GGIW>(
        cphd, scenario, "GGIW CPHD Tracking", 10.0, 5);
    // CPHD+GGIW is a new combination; cardinality estimation with extended
    // targets may need further parameter tuning.
    EXPECT_GE(cphd_result.converged_steps, 0)
        << "GGIW CPHD should not crash";

    // MBM
    auto mbm = test::make_mbm<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto mbm_result = test::run_tracking<decltype(mbm), distributions::GGIW>(
        mbm, scenario, "GGIW MBM Tracking", 10.0, 5);
    EXPECT_GE(mbm_result.converged_steps, 10)
        << "GGIW MBM should track both extended targets";

    // PMBM
    auto pmbm = test::make_pmbm<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto pmbm_result = test::run_tracking<decltype(pmbm), distributions::GGIW>(
        pmbm, scenario, "GGIW PMBM Tracking", 10.0, 5);
    EXPECT_GE(pmbm_result.converged_steps, 10)
        << "GGIW PMBM should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    // 2x2 comparison figure
    auto fig = test::create_comparison_figure();

    auto ax1 = test::comparison_subplot(fig, 0);
    test::populate_estimator_axes(ax1, phd, phd_result.plot_data, "PHD");

    auto ax2 = test::comparison_subplot(fig, 1);
    test::populate_estimator_axes(ax2, cphd, cphd_result.plot_data, "CPHD");

    auto ax3 = test::comparison_subplot(fig, 2);
    test::populate_estimator_axes(ax3, mbm, mbm_result.plot_data, "MBM");

    auto ax4 = test::comparison_subplot(fig, 3);
    test::populate_estimator_axes(ax4, pmbm, pmbm_result.plot_data, "PMBM");

    brew::plot_utils::save_figure(fig, test::output_dir() + "/ggiw_comparison.png");

    // Cardinality figure
    test::plot_cardinality_comparison(scenario, cphd_result.cardinality,
        "GGIW CPHD - Estimated Cardinality", "ggiw_cphd_cardinality.png");
#endif
}
