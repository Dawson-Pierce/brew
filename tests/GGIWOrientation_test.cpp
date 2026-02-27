#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"

using namespace brew;

class GGIWOrientationTracking : public ::testing::Test {
protected:
    test::ScenarioData scenario;
    test::ScenarioParams params;

    void SetUp() override {
        scenario = test::make_base_scenario();
        test::generate_scenario_extended_measurements(scenario);
        params = test::make_default_params(scenario);
    }
};

TEST_F(GGIWOrientationTracking, Comparison) {
    // PHD
    auto phd = test::make_phd<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto phd_result = test::run_tracking<decltype(phd), models::GGIWOrientation>(
        phd, scenario, "GGIWOrientation PHD Tracking", 10.0, 5);
    EXPECT_GE(phd_result.converged_steps, 10)
        << "GGIWOrientation PHD should track both extended targets";

    // CPHD
    auto cphd = test::make_cphd<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto cphd_result = test::run_tracking_cphd<decltype(cphd), models::GGIWOrientation>(
        cphd, scenario, "GGIWOrientation CPHD Tracking", 10.0, 5);
    EXPECT_GE(cphd_result.converged_steps, 0)
        << "GGIWOrientation CPHD should not crash";

    // MBM
    auto mbm = test::make_mbm<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto mbm_result = test::run_tracking<decltype(mbm), models::GGIWOrientation>(
        mbm, scenario, "GGIWOrientation MBM Tracking", 10.0, 5);
    EXPECT_GE(mbm_result.converged_steps, 10)
        << "GGIWOrientation MBM should track both extended targets";

    // PMBM
    auto pmbm = test::make_pmbm<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto pmbm_result = test::run_tracking<decltype(pmbm), models::GGIWOrientation>(
        pmbm, scenario, "GGIWOrientation PMBM Tracking", 10.0, 5);
    EXPECT_GE(pmbm_result.converged_steps, 10)
        << "GGIWOrientation PMBM should track both extended targets";

    // MB
    auto mb = test::make_mb<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto mb_result = test::run_tracking<decltype(mb), models::GGIWOrientation>(
        mb, scenario, "GGIWOrientation MB Tracking", 10.0, 5);
    EXPECT_GE(mb_result.converged_steps, 10)
        << "GGIWOrientation MB should track both extended targets";

    // LMB
    auto lmb = test::make_lmb<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto lmb_result = test::run_tracking<decltype(lmb), models::GGIWOrientation>(
        lmb, scenario, "GGIWOrientation LMB Tracking", 10.0, 5);
    EXPECT_GE(lmb_result.converged_steps, 10)
        << "GGIWOrientation LMB should track both extended targets";

    // GLMB
    auto glmb = test::make_glmb<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto glmb_result = test::run_tracking<decltype(glmb), models::GGIWOrientation>(
        glmb, scenario, "GGIWOrientation GLMB Tracking", 10.0, 5);
    EXPECT_GE(glmb_result.converged_steps, 10)
        << "GGIWOrientation GLMB should track both extended targets";

    // JGLMB
    auto jglmb = test::make_jglmb<models::GGIWOrientation>(
        test::make_ggiw_orientation_ekf(scenario), test::make_ggiw_orientation_birth(0.1), params);
    auto jglmb_result = test::run_tracking<decltype(jglmb), models::GGIWOrientation>(
        jglmb, scenario, "GGIWOrientation JGLMB Tracking", 10.0, 5);
    EXPECT_GE(jglmb_result.converged_steps, 10)
        << "GGIWOrientation JGLMB should track both extended targets";

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

    brew::plot_utils::save_figure(fig, test::output_dir() + "/ggiw_orientation_comparison.png");

    // Cardinality figure
    test::plot_cardinality_comparison(scenario, cphd_result.cardinality,
        "GGIWOrientation CPHD - Estimated Cardinality", "ggiw_orientation_cphd_cardinality.png");
#endif
}
