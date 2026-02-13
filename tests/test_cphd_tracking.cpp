#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"
#include "brew/multi_target/cphd.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

// ---- CPHD Gaussian tracking: Two Point Targets ----

TEST(CPHDTracking, TwoPointTargets) {
    auto scenario = test::make_two_point_targets_scenario();

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

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.05);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.05);

    auto intensity = std::make_unique<distributions::Mixture<distributions::Gaussian>>();

    multi_target::CPHD<distributions::Gaussian> cphd;
    cphd.set_filter(std::move(ekf));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::move(intensity));
    cphd.set_prob_detection(scenario.p_detect);
    cphd.set_prob_survive(0.99);
    cphd.set_clutter_rate(1.0);
    cphd.set_clutter_density(1e-4);
    cphd.set_prune_threshold(1e-5);
    cphd.set_merge_threshold(4.0);
    cphd.set_max_components(30);
    cphd.set_extract_threshold(0.4);
    cphd.set_gate_threshold(25.0);
    cphd.set_poisson_cardinality(0.1);
    cphd.set_poisson_birth_cardinality(0.1);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
    std::vector<double> est_card_vec;
#endif

    test::print_tracking_header("CPHD Tracking - Two Point Targets");
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err_A"
              << std::setw(12) << "Err_B"
              << std::setw(12) << "Est_Card"
              << "\n";
    std::cout << std::string(71, '-') << "\n";

    int converged_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        cphd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        cphd.correct(meas);
        cphd.cleanup();

        const auto& extracted = cphd.extracted_mixtures();
        const auto& latest = *extracted.back();

        Eigen::VectorXd truth_a = scenario.targets[0].states[k].head(2);
        Eigen::VectorXd truth_b = scenario.targets[1].states[k].head(2);

        double err_a = test::closest_estimate_error(latest, truth_a);
        double err_b = test::closest_estimate_error(latest, truth_b);
        double est_card = cphd.estimated_cardinality();

        std::cout << std::setw(5) << k
                  << std::setw(10) << meas.cols()
                  << std::setw(10) << cphd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_a
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_b
                  << std::setw(12) << std::fixed << std::setprecision(2) << est_card
                  << "\n";

        if (k >= 10 && latest.size() >= 2) {
            if (err_a < 5.0 && err_b < 5.0) {
                converged_steps++;
            }
        }

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
        est_card_vec.push_back(est_card);
#endif
    }

    std::cout << "\nConverged steps (k>=10, both errors < 5.0): "
              << converged_steps << " / " << (scenario.num_steps - 10) << "\n";

    EXPECT_GE(converged_steps, 10) << "CPHD should track both targets for most of the run";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000); fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        test::plot_common_elements(ax, plot_data);
        test::plot_all_extracted(ax, cphd.extracted_mixtures());

        ax->title("CPHD - Two Point Targets");
        ax->xlabel("x");
        ax->ylabel("y");
        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/cphd_gaussian_two_targets.png");
    }

    // Cardinality plot
    {
        auto fig2 = matplot::figure(true);
        fig2->width(800);
        fig2->height(400);
        auto ax2 = fig2->current_axes();
        ax2->hold(true);

        std::vector<double> steps_vec, true_card_vec;
        for (int k = 0; k < scenario.num_steps; ++k) {
            steps_vec.push_back(static_cast<double>(k));
            true_card_vec.push_back(2.0);
        }

        auto tc = ax2->plot(steps_vec, true_card_vec, "--");
        tc->color({0.f, 0.f, 0.f, 0.f});
        tc->line_width(2.0f);

        auto ec = ax2->plot(steps_vec, est_card_vec);
        ec->color({0.f, 0.f, 0.4470f, 0.7410f});
        ec->line_width(2.0f);

        ax2->title("CPHD - Estimated Cardinality");
        ax2->xlabel("Time step");
        ax2->ylabel("Cardinality");

        brew::plot_utils::save_figure(fig2,
            "/workspace/brew/output/cphd_cardinality_two_targets.png");
    }
#endif
}

// ---- CPHD with clutter and dropouts ----

TEST(CPHDTracking, WithClutterAndDropouts) {
    auto scenario = test::make_clutter_scenario();

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
    birth_cov(0, 0) = 250.0; birth_cov(1, 1) = 250.0;
    birth_cov(2, 2) = 15.0; birth_cov(3, 3) = 15.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << -15.0, -10.0, 0.0, 0.0;
    b2 << 55.0, 5.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.05);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.05);

    multi_target::CPHD<distributions::Gaussian> cphd;
    cphd.set_filter(std::move(ekf));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::make_unique<distributions::Mixture<distributions::Gaussian>>());
    cphd.set_prob_detection(scenario.p_detect);
    cphd.set_prob_survive(0.99);
    cphd.set_clutter_rate(scenario.clutter_rate);
    cphd.set_clutter_density(1.0 / scenario.surveillance_area);
    cphd.set_prune_threshold(1e-5);
    cphd.set_merge_threshold(4.0);
    cphd.set_max_components(50);
    cphd.set_extract_threshold(0.4);
    cphd.set_gate_threshold(25.0);
    cphd.set_poisson_cardinality(0.1);
    cphd.set_poisson_birth_cardinality(0.1);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
    std::vector<double> est_card_vec;
#endif

    test::print_tracking_header("CPHD Tracking - Clutter & Dropouts");
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err_A"
              << std::setw(12) << "Err_B"
              << std::setw(12) << "Est_Card"
              << "\n";
    std::cout << std::string(71, '-') << "\n";

    int converged_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        cphd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        cphd.correct(meas);
        cphd.cleanup();

        const auto& extracted = cphd.extracted_mixtures();
        const auto& latest = *extracted.back();

        Eigen::VectorXd truth_a = scenario.targets[0].states[k].head(2);
        Eigen::VectorXd truth_b = scenario.targets[1].states[k].head(2);

        double err_a = test::closest_estimate_error(latest, truth_a);
        double err_b = test::closest_estimate_error(latest, truth_b);
        double est_card = cphd.estimated_cardinality();

        std::cout << std::setw(5) << k
                  << std::setw(10) << meas.cols()
                  << std::setw(10) << cphd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_a
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_b
                  << std::setw(12) << std::fixed << std::setprecision(2) << est_card
                  << "\n";

        if (k >= 10 && latest.size() >= 2) {
            if (err_a < 8.0 && err_b < 8.0) {
                converged_steps++;
            }
        }

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
        est_card_vec.push_back(est_card);
#endif
    }

    std::cout << "\nConverged steps (k>=10, both errors < 8.0): "
              << converged_steps << " / " << (scenario.num_steps - 10) << "\n";

    EXPECT_GE(converged_steps, 5) << "CPHD should track targets under clutter";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000); fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        test::plot_common_elements(ax, plot_data);
        test::plot_all_extracted(ax, cphd.extracted_mixtures());

        ax->title("CPHD - Clutter & Dropouts");
        ax->xlabel("x");
        ax->ylabel("y");
        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/cphd_gaussian_clutter.png");
    }

    // Cardinality plot
    {
        auto fig2 = matplot::figure(true);
        fig2->width(800);
        fig2->height(400);
        auto ax2 = fig2->current_axes();
        ax2->hold(true);

        std::vector<double> steps_vec, true_card_vec;
        for (int k = 0; k < scenario.num_steps; ++k) {
            steps_vec.push_back(static_cast<double>(k));
            true_card_vec.push_back(2.0);
        }

        auto tc = ax2->plot(steps_vec, true_card_vec, "--");
        tc->color({0.f, 0.f, 0.f, 0.f});
        tc->line_width(2.0f);

        auto ec = ax2->plot(steps_vec, est_card_vec);
        ec->color({0.f, 0.f, 0.4470f, 0.7410f});
        ec->line_width(2.0f);

        ax2->title("CPHD - Estimated Cardinality (Clutter)");
        ax2->xlabel("Time step");
        ax2->ylabel("Cardinality");

        brew::plot_utils::save_figure(fig2,
            "/workspace/brew/output/cphd_cardinality_clutter.png");
    }
#endif
}

// ---- Variable number of targets ----

TEST(CPHDTracking, VariableTargetCount) {
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

    multi_target::CPHD<distributions::Gaussian> cphd;
    cphd.set_filter(std::move(ekf));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::make_unique<distributions::Mixture<distributions::Gaussian>>());
    cphd.set_prob_detection(scenario.p_detect);
    cphd.set_prob_survive(0.99);
    cphd.set_clutter_rate(1.0);
    cphd.set_clutter_density(1e-4);
    cphd.set_prune_threshold(1e-5);
    cphd.set_merge_threshold(4.0);
    cphd.set_max_components(50);
    cphd.set_extract_threshold(0.4);
    cphd.set_gate_threshold(25.0);
    cphd.set_poisson_cardinality(0.1);
    cphd.set_poisson_birth_cardinality(0.09);

#ifdef BREW_ENABLE_PLOTTING
    std::vector<double> est_card_vec;
    std::vector<double> true_card_vec;
#endif

    test::print_tracking_header("CPHD Tracking - Variable Target Count");
    std::cout << std::setw(5) << "Step"
              << std::setw(12) << "True_Card"
              << std::setw(12) << "Est_Card"
              << std::setw(10) << "N_est"
              << "\n";
    std::cout << std::string(39, '-') << "\n";

    for (int k = 0; k < scenario.num_steps; ++k) {
        cphd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        cphd.correct(meas);
        cphd.cleanup();

        int true_card = scenario.true_cardinality(k);
        double est_card = cphd.estimated_cardinality();
        const auto& latest = *cphd.extracted_mixtures().back();

        std::cout << std::setw(5) << k
                  << std::setw(12) << true_card
                  << std::setw(12) << std::fixed << std::setprecision(2) << est_card
                  << std::setw(10) << latest.size()
                  << "\n";

#ifdef BREW_ENABLE_PLOTTING
        true_card_vec.push_back(static_cast<double>(true_card));
        est_card_vec.push_back(est_card);
#endif
    }

    double final_est = cphd.estimated_cardinality();
    EXPECT_GT(final_est, 0.0) << "Should estimate at least some targets";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(800);
        fig->height(400);
        auto ax = fig->current_axes();
        ax->hold(true);

        std::vector<double> steps_vec;
        for (int k = 0; k < scenario.num_steps; ++k) {
            steps_vec.push_back(static_cast<double>(k));
        }

        auto tc = ax->plot(steps_vec, true_card_vec, "--");
        tc->color({0.f, 0.f, 0.f, 0.f});
        tc->line_width(2.0f);

        auto ec = ax->plot(steps_vec, est_card_vec);
        ec->color({0.f, 0.f, 0.4470f, 0.7410f});
        ec->line_width(2.0f);

        ax->title("CPHD - Variable Target Count");
        ax->xlabel("Time step");
        ax->ylabel("Cardinality");

        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/cphd_cardinality_variable.png");
    }
#endif
}
