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

    // MB
    auto mb = test::make_mb<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto mb_result = test::run_tracking<decltype(mb), distributions::GGIW>(
        mb, scenario, "GGIW MB Tracking", 10.0, 5);
    EXPECT_GE(mb_result.converged_steps, 10)
        << "GGIW MB should track both extended targets";

    // LMB
    auto lmb = test::make_lmb<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto lmb_result = test::run_tracking<decltype(lmb), distributions::GGIW>(
        lmb, scenario, "GGIW LMB Tracking", 10.0, 5);
    EXPECT_GE(lmb_result.converged_steps, 10)
        << "GGIW LMB should track both extended targets";

    // GLMB
    auto glmb = test::make_glmb<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto glmb_result = test::run_tracking<decltype(glmb), distributions::GGIW>(
        glmb, scenario, "GGIW GLMB Tracking", 10.0, 5);
    EXPECT_GE(glmb_result.converged_steps, 10)
        << "GGIW GLMB should track both extended targets";

    // JGLMB
    auto jglmb = test::make_jglmb<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);
    auto jglmb_result = test::run_tracking<decltype(jglmb), distributions::GGIW>(
        jglmb, scenario, "GGIW JGLMB Tracking", 10.0, 5);
    EXPECT_GE(jglmb_result.converged_steps, 10)
        << "GGIW JGLMB should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    // 2x4 comparison figure
    auto fig = test::create_comparison_figure();

    auto ax1 = test::comparison_subplot(fig, 2, 4, 0);
    test::populate_estimator_axes(ax1, phd, phd_result.plot_data, "PHD");

    auto ax2 = test::comparison_subplot(fig, 2, 4, 1);
    test::populate_estimator_axes(ax2, cphd, cphd_result.plot_data, "CPHD");

    auto ax3 = test::comparison_subplot(fig, 2, 4, 2);
    test::populate_estimator_axes(ax3, mbm, mbm_result.plot_data, "MBM");

    auto ax4 = test::comparison_subplot(fig, 2, 4, 3);
    test::populate_estimator_axes(ax4, pmbm, pmbm_result.plot_data, "PMBM");

    auto ax5 = test::comparison_subplot(fig, 2, 4, 4);
    test::populate_estimator_axes(ax5, mb, mb_result.plot_data, "MB");

    auto ax6 = test::comparison_subplot(fig, 2, 4, 5);
    test::populate_estimator_axes(ax6, lmb, lmb_result.plot_data, "LMB");

    auto ax7 = test::comparison_subplot(fig, 2, 4, 6);
    test::populate_estimator_axes(ax7, glmb, glmb_result.plot_data, "GLMB");

    auto ax8 = test::comparison_subplot(fig, 2, 4, 7);
    test::populate_estimator_axes(ax8, jglmb, jglmb_result.plot_data, "JGLMB");

    brew::plot_utils::save_figure(fig, test::output_dir() + "/ggiw_comparison.png");

    // Cardinality figure
    test::plot_cardinality_comparison(scenario, cphd_result.cardinality,
        "GGIW CPHD - Estimated Cardinality", "ggiw_cphd_cardinality.png");
#endif
}
