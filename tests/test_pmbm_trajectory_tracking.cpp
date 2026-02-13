#include <gtest/gtest.h>
#include "tracking_test_utils.hpp"
#include "brew/multi_target/pmbm.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"

using namespace brew;

// PMBM inherently provides trajectory extraction through its Bernoulli track
// structure, so it does not need trajectory distribution types (TrajectoryGaussian
// or TrajectoryGGIW). This file tests PMBM's trajectory tracking capability
// using regular distribution types.

// ---- PMBM trajectory tracking: Two Point Targets ----

TEST(PMBMTrajectoryTracking, TwoPointTargets) {
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
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.1);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.1);

    auto poisson = std::make_unique<distributions::Mixture<distributions::Gaussian>>();

    multi_target::PMBM<distributions::Gaussian> pmbm;
    pmbm.set_filter(std::move(ekf));
    pmbm.set_birth_model(std::move(birth));
    pmbm.set_poisson_intensity(std::move(poisson));
    pmbm.set_prob_detection(scenario.p_detect);
    pmbm.set_prob_survive(0.99);
    pmbm.set_clutter_rate(1.0);
    pmbm.set_clutter_density(1e-4);
    pmbm.set_prune_poisson_threshold(1e-4);
    pmbm.set_merge_poisson_threshold(4.0);
    pmbm.set_max_poisson_components(100);
    pmbm.set_prune_threshold_hypothesis(1e-4);
    pmbm.set_prune_threshold_bernoulli(1e-3);
    pmbm.set_recycle_threshold(0.1);
    pmbm.set_max_hypotheses(50);
    pmbm.set_extract_threshold(0.4);
    pmbm.set_gate_threshold(25.0);
    pmbm.set_k_best(5);

#ifdef BREW_ENABLE_PLOTTING
    test::TrackingPlotData plot_data(static_cast<int>(scenario.targets.size()));
#endif

    test::print_tracking_header("PMBM Trajectory Tracking - Two Point Targets");
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_hyp"
              << std::setw(10) << "N_trk"
              << std::setw(10) << "N_est"
              << std::setw(10) << "N_poi"
              << std::setw(12) << "Err_A"
              << std::setw(12) << "Err_B"
              << "\n";
    std::cout << std::string(79, '-') << "\n";

    int converged_steps = 0;

    for (int k = 0; k < scenario.num_steps; ++k) {
        pmbm.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        pmbm.correct(meas);
        pmbm.cleanup();

        const auto& extracted = pmbm.extracted_mixtures();
        const auto& latest = *extracted.back();

        Eigen::VectorXd truth_a = scenario.targets[0].states[k].head(2);
        Eigen::VectorXd truth_b = scenario.targets[1].states[k].head(2);

        double err_a = test::closest_estimate_error(latest, truth_a);
        double err_b = test::closest_estimate_error(latest, truth_b);

        std::cout << std::setw(5) << k
                  << std::setw(10) << meas.cols()
                  << std::setw(10) << pmbm.global_hypotheses().size()
                  << std::setw(10) << pmbm.num_tracks()
                  << std::setw(10) << latest.size()
                  << std::setw(10) << pmbm.poisson_intensity().size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_a
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_b
                  << "\n";

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
        << "PMBM should track both targets via inherent trajectory extraction";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000); fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        test::plot_common_elements(ax, plot_data);
        test::plot_track_histories(ax, pmbm.track_histories());
        test::plot_final_extracted(ax, pmbm.extracted_mixtures());

        ax->title("PMBM Trajectory Tracking - Two Point Targets");
        ax->xlabel("x");
        ax->ylabel("y");
        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/pmbm_traj_gaussian_two_targets.png");
    }
#endif
}
