#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"
#include "brew/multi_target/phd.hpp"
#include "brew/filters/trajectory_gaussian_ekf.hpp"
#include "brew/filters/trajectory_ggiw_ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

// ---- Trajectory Gaussian PHD tracking test ----

TEST(PHDTrajectoryTracking, TwoPointTargets) {
    auto scenario = test::make_two_point_targets_scenario();

    const int l_window = 10;
    auto dyn = std::make_shared<dynamics::Integrator2D>();

    auto ekf = std::make_unique<filters::TrajectoryGaussianEKF>();
    ekf->set_dynamics(dyn);
    ekf->set_window_size(l_window);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::TrajectoryGaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 100.0; birth_cov(1, 1) = 100.0;
    birth_cov(2, 2) = 10.0; birth_cov(3, 3) = 10.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<distributions::TrajectoryGaussian>(0, 4, b1, birth_cov), 0.05);
    birth->add_component(
        std::make_unique<distributions::TrajectoryGaussian>(0, 4, b2, birth_cov), 0.05);

    auto intensity = std::make_unique<distributions::Mixture<distributions::TrajectoryGaussian>>();

    multi_target::PHD<distributions::TrajectoryGaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(scenario.p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(30);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
#endif

    test::print_tracking_header("Trajectory Gaussian PHD - Two Point Targets");
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err_A"
              << std::setw(12) << "Err_B"
              << "\n";
    std::cout << std::string(59, '-') << "\n";

    int converged_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        phd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        Eigen::VectorXd truth_a = scenario.targets[0].states[k].head(2);
        Eigen::VectorXd truth_b = scenario.targets[1].states[k].head(2);

        double err_a = test::closest_estimate_error(latest, truth_a);
        double err_b = test::closest_estimate_error(latest, truth_b);

        test::print_tracking_step(k, static_cast<int>(meas.cols()),
            static_cast<int>(phd.intensity().size()),
            static_cast<int>(latest.size()), err_a, err_b);

        if (k >= 10 && latest.size() >= 2) {
            if (err_a < 5.0 && err_b < 5.0) {
                converged_steps++;
            }
        }

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
#endif
    }

    std::cout << "\nConverged steps (k>=10, both errors < 5.0): "
              << converged_steps << " / " << (scenario.num_steps - 10) << "\n";

    EXPECT_GE(converged_steps, 10)
        << "Trajectory Gaussian PHD should track both targets";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000); fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        test::plot_common_elements(ax, plot_data);
        test::plot_final_extracted(ax, phd.extracted_mixtures());

        ax->title("Trajectory Gaussian PHD - Two Point Targets");
        ax->xlabel("x");
        ax->ylabel("y");
        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/phd_traj_gaussian_two_targets.png");
    }
#endif
}

// ---- Trajectory GGIW PHD tracking test ----

TEST(PHDTrajectoryTracking, ExtendedTargetGGIW) {
    auto scenario = test::make_extended_target_scenario();

    const int l_window = 10;
    auto dyn = std::make_shared<dynamics::Integrator2D>();

    auto ggiw_ekf = std::make_unique<filters::TrajectoryGGIWEKF>();
    ggiw_ekf->set_dynamics(dyn);
    ggiw_ekf->set_window_size(l_window);
    ggiw_ekf->set_temporal_decay(1.0);
    ggiw_ekf->set_forgetting_factor(5.0);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ggiw_ekf->set_measurement_jacobian(H);
    ggiw_ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ggiw_ekf->set_measurement_noise(scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::TrajectoryGGIW>>();
    Eigen::VectorXd b_mean(4);
    b_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd b_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 20.0 * Eigen::MatrixXd::Identity(2, 2);
    birth->add_component(
        std::make_unique<distributions::TrajectoryGGIW>(
            0, 4, b_mean, b_cov, 10.0, 1.0, 10.0, b_V), 0.1);

    auto intensity = std::make_unique<distributions::Mixture<distributions::TrajectoryGGIW>>();

    multi_target::PHD<distributions::TrajectoryGGIW> phd;
    phd.set_filter(std::move(ggiw_ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(scenario.p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(20);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);
    phd.set_extended_target(true);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
#endif

    test::print_tracking_header("Trajectory GGIW PHD - Extended Target");
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err"
              << "\n";
    std::cout << std::string(47, '-') << "\n";

    int tracked_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        phd.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        Eigen::VectorXd truth_pos = scenario.targets[0].states[k].head(2);
        double err = test::closest_estimate_error(latest, truth_pos);

        test::print_tracking_step(k, static_cast<int>(meas.cols()),
            static_cast<int>(phd.intensity().size()),
            static_cast<int>(latest.size()), err);

        if (k >= 5 && latest.size() >= 1 && err < 10.0) {
            tracked_steps++;
        }

#ifdef BREW_ENABLE_PLOTTING
        test::accumulate_plot_step(plot_data, scenario, k, meas, latest);
#endif
    }

    std::cout << "\nTracked steps (k>=5, error < 10.0): "
              << tracked_steps << " / " << (scenario.num_steps - 5) << "\n";

    EXPECT_GE(tracked_steps, 5)
        << "Trajectory GGIW PHD should track the extended target";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000); fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        test::plot_common_elements(ax, plot_data);
        test::plot_final_extracted(ax, phd.extracted_mixtures());

        ax->title("Trajectory GGIW PHD - Extended Target");
        ax->xlabel("x");
        ax->ylabel("y");
        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/phd_traj_ggiw_extended_target.png");
    }
#endif
}
