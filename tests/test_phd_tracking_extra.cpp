#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"
#include "brew/multi_target/phd.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

// ---- GM-PHD with clutter and dropouts ----

TEST(PHDTracking, GMWithClutterAndDropouts) {
    auto scenario = test::make_clutter_scenario();

    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::EKF>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.8 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 250.0; birth_cov(1, 1) = 250.0;
    birth_cov(2, 2) = 15.0; birth_cov(3, 3) = 15.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << -15.0, -10.0, 0.0, 0.0;
    b2 << 55.0, 5.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.08);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.08);

    auto intensity = std::make_unique<distributions::Mixture<distributions::Gaussian>>();

    multi_target::PHD<distributions::Gaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(scenario.p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(scenario.clutter_rate);
    phd.set_clutter_density(1.0 / scenario.surveillance_area);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(40);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
#endif

    int good_steps = 0;
    int eval_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        phd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        if (k >= 12) {
            eval_steps++;
            Eigen::VectorXd truth_a = scenario.targets[0].states[k].head(2);
            Eigen::VectorXd truth_b = scenario.targets[1].states[k].head(2);
            double err_a = test::closest_estimate_error(latest, truth_a);
            double err_b = test::closest_estimate_error(latest, truth_b);
            if (latest.size() >= 2 && err_a < 7.0 && err_b < 7.0) {
                good_steps++;
            }
        }

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
#endif
    }

    EXPECT_GE(good_steps, 8) << "GM-PHD should maintain both tracks despite clutter";
    EXPECT_GE(eval_steps, 1);

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000); fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        test::plot_common_elements(ax, plot_data);
        test::plot_all_extracted(ax, phd.extracted_mixtures());

        ax->title("GM-PHD - Clutter and Dropouts");
        ax->xlabel("x");
        ax->ylabel("y");
        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/phd_gaussian_clutter.png");
    }
#endif
}

// ---- GM-PHD with variable target count ----

TEST(PHDTracking, VariableTargetCount) {
    auto scenario = test::make_variable_targets_scenario();

    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::EKF>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 100.0; birth_cov(1, 1) = 100.0;
    birth_cov(2, 2) = 10.0; birth_cov(3, 3) = 10.0;

    Eigen::VectorXd b1(4), b2(4), b3(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    b3 << 25.0, 30.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.03);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.03);
    birth->add_component(std::make_unique<distributions::Gaussian>(b3, birth_cov), 0.03);

    auto intensity = std::make_unique<distributions::Mixture<distributions::Gaussian>>();

    multi_target::PHD<distributions::Gaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(scenario.p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(50);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);

    test::print_tracking_header("GM-PHD Tracking - Variable Target Count");
    std::cout << std::setw(5) << "Step"
              << std::setw(12) << "True_Card"
              << std::setw(10) << "N_est"
              << std::setw(10) << "N_meas"
              << "\n";
    std::cout << std::string(37, '-') << "\n";

    int good_card_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        phd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        int true_card = scenario.true_cardinality(k);

        std::cout << std::setw(5) << k
                  << std::setw(12) << true_card
                  << std::setw(10) << latest.size()
                  << std::setw(10) << meas.cols()
                  << "\n";

        if (k >= 10 && static_cast<int>(latest.size()) >= true_card - 1
            && static_cast<int>(latest.size()) <= true_card + 1) {
            good_card_steps++;
        }
    }

    EXPECT_GE(good_card_steps, 5) << "PHD should roughly track variable cardinality";
}
