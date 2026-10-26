#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"

using namespace brew;

class GMTracking : public ::testing::Test {
protected:
    test::ScenarioData scenario;
    test::ScenarioParams params;

    void SetUp() override {
        scenario = test::make_base_scenario();
        test::generate_scenario_point_measurements(scenario);
        params = test::make_default_params(scenario);
    }
};

TEST_F(GMTracking, Comparison) {

    auto phd = test::make_phd<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.05), params);
    auto phd_result = test::run_tracking<decltype(phd), models::Gaussian<>>(
        phd, scenario, "GM PHD Tracking", 5.0, 10);
    EXPECT_GE(phd_result.converged_steps, 10)
        << "PHD should track both targets for most of the run";

    auto cphd = test::make_cphd<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.05), params);
    auto cphd_result = test::run_tracking_cphd<decltype(cphd), models::Gaussian<>>(
        cphd, scenario, "GM CPHD Tracking", 5.0, 10);
    EXPECT_GE(cphd_result.converged_steps, 10)
        << "CPHD should track both targets for most of the run";

    auto mbm = test::make_mbm<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.1), params);
    auto mbm_result = test::run_tracking<decltype(mbm), models::Gaussian<>>(
        mbm, scenario, "GM MBM Tracking", 5.0, 10);
    EXPECT_GE(mbm_result.converged_steps, 10)
        << "MBM should track both targets for most of the run";

    auto pmbm = test::make_pmbm<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.1), params);
    auto pmbm_result = test::run_tracking<decltype(pmbm), models::Gaussian<>>(
        pmbm, scenario, "GM PMBM Tracking", 5.0, 10);
    EXPECT_GE(pmbm_result.converged_steps, 10)
        << "PMBM should track both targets for most of the run";

    auto glmb = test::make_glmb<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.1), params);
    auto glmb_result = test::run_tracking<decltype(glmb), models::Gaussian<>>(
        glmb, scenario, "GM GLMB Tracking", 5.0, 10);
    EXPECT_GE(glmb_result.converged_steps, 10)
        << "GLMB should track both targets for most of the run";

    auto jglmb = test::make_jglmb<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.1), params);
    auto jglmb_result = test::run_tracking<decltype(jglmb), models::Gaussian<>>(
        jglmb, scenario, "GM JGLMB Tracking", 5.0, 10);
    EXPECT_GE(jglmb_result.converged_steps, 10)
        << "JGLMB should track both targets for most of the run";

#ifdef BREW_ENABLE_PLOTTING

    auto fig = test::create_comparison_figure();

    auto ax1 = test::comparison_subplot(fig, 2, 3, 0);
    test::populate_estimator_axes(ax1, phd, phd_result.plot_data, "PHD");

    auto ax2 = test::comparison_subplot(fig, 2, 3, 1);
    test::populate_estimator_axes(ax2, cphd, cphd_result.plot_data, "CPHD");

    auto ax3 = test::comparison_subplot(fig, 2, 3, 2);
    test::populate_estimator_axes(ax3, mbm, mbm_result.plot_data, "MBM");

    auto ax4 = test::comparison_subplot(fig, 2, 3, 3);
    test::populate_estimator_axes(ax4, pmbm, pmbm_result.plot_data, "PMBM");

    auto ax5 = test::comparison_subplot(fig, 2, 3, 4);
    test::populate_estimator_axes(ax5, glmb, glmb_result.plot_data, "GLMB");

    auto ax6 = test::comparison_subplot(fig, 2, 3, 5);
    test::populate_estimator_axes(ax6, jglmb, jglmb_result.plot_data, "JGLMB");

    brew::plot_utils::save_figure(fig, test::output_dir() + "/gm_comparison.png");

    test::plot_cardinality_comparison(scenario, cphd_result.cardinality,
        "GM CPHD - Estimated Cardinality", "gm_cphd_cardinality.png");
#endif
}

TEST_F(GMTracking, PHDWithUKF) {

    auto phd = test::make_phd<models::Gaussian<>>(
        test::make_ukf(scenario), test::make_gm_birth(0.05), params);
    auto result = test::run_tracking<decltype(phd), models::Gaussian<>>(
        phd, scenario, "GM PHD-UKF Tracking", 5.0, 10);
    EXPECT_GE(result.converged_steps, 10)
        << "UKF-driven PHD should track both targets for most of the run";
}

TEST_F(GMTracking, JGLMBGibbs) {

    auto jglmb = test::make_jglmb<models::Gaussian<>>(
        test::make_ekf(scenario), test::make_gm_birth(0.1), params);
    jglmb.set_use_gibbs(true);
    auto result = test::run_tracking<decltype(jglmb), models::Gaussian<>>(
        jglmb, scenario, "GM JGLMB-Gibbs Tracking", 5.0, 10);
    EXPECT_GE(result.converged_steps, 10)
        << "JGLMB with Gibbs sampling should track both targets for most of the run";
}
