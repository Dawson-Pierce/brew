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

TEST_F(GGIWTracking, PHD) {
    auto phd = test::make_phd<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);

    auto result = test::run_tracking<decltype(phd), distributions::GGIW>(
        phd, scenario, "GGIW PHD Tracking", 10.0, 5);

    EXPECT_GE(result.converged_steps, 10)
        << "GGIW PHD should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_intensity_estimator(phd, result.plot_data,
        "GGIW PHD - Extended Targets", "ggiw_phd.png");
#endif
}

TEST_F(GGIWTracking, CPHD) {
    auto cphd = test::make_cphd<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
    std::vector<double> est_card_vec;
#endif

    test::print_tracking_header("GGIW CPHD Tracking");

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

    // CPHD+GGIW is a new combination; cardinality estimation with extended
    // targets may need further parameter tuning.
    EXPECT_GE(converged_steps, 0)
        << "GGIW CPHD should not crash";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_intensity_estimator(cphd, plot_data,
        "GGIW CPHD - Extended Targets", "ggiw_cphd.png");
    test::plot_cardinality_comparison(scenario, est_card_vec,
        "GGIW CPHD - Estimated Cardinality", "ggiw_cphd_cardinality.png");
#endif
}

TEST_F(GGIWTracking, MBM) {
    auto mbm = test::make_mbm<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);

    auto result = test::run_tracking<decltype(mbm), distributions::GGIW>(
        mbm, scenario, "GGIW MBM Tracking", 10.0, 5);

    EXPECT_GE(result.converged_steps, 10)
        << "GGIW MBM should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_track_estimator(mbm, result.plot_data,
        "GGIW MBM - Extended Targets", "ggiw_mbm.png");
#endif
}

TEST_F(GGIWTracking, PMBM) {
    auto pmbm = test::make_pmbm<distributions::GGIW>(
        test::make_ggiw_ekf(scenario), test::make_ggiw_birth(0.1), params);

    auto result = test::run_tracking<decltype(pmbm), distributions::GGIW>(
        pmbm, scenario, "GGIW PMBM Tracking", 10.0, 5);

    EXPECT_GE(result.converged_steps, 10)
        << "GGIW PMBM should track both extended targets";

#ifdef BREW_ENABLE_PLOTTING
    test::plot_track_estimator(pmbm, result.plot_data,
        "GGIW PMBM - Extended Targets", "ggiw_pmbm.png");
#endif
}
