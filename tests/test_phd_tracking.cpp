#include <gtest/gtest.h>
#include "brew/multi_target/phd.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/plot_utils/plot_options.hpp>
#include <brew/plot_utils/plot_gaussian.hpp>
#include <brew/plot_utils/plot_ggiw.hpp>
#include <brew/plot_utils/color_palette.hpp>
#include <filesystem>
#endif

using namespace brew;

// ---- Helpers for generating truth and measurements ----

struct TruthTarget {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states; // state at each timestep from birth
};

// Generate a linear truth trajectory using dynamics
static TruthTarget make_linear_target(
    const Eigen::VectorXd& initial_state,
    int birth_time, int death_time, double dt,
    dynamics::DynamicsBase& dyn)
{
    TruthTarget t;
    t.birth_time = birth_time;
    t.death_time = death_time;
    Eigen::VectorXd x = initial_state;
    for (int k = birth_time; k <= death_time; ++k) {
        t.states.push_back(x);
        x = dyn.propagate_state(dt, x);
    }
    return t;
}

// Generate noisy position measurements from truth targets at a given timestep
static Eigen::MatrixXd generate_measurements(
    const std::vector<TruthTarget>& targets,
    int timestep,
    double meas_std,
    std::mt19937& rng,
    double p_detect = 0.95)
{
    std::normal_distribution<double> noise(0.0, meas_std);
    std::uniform_real_distribution<double> det_roll(0.0, 1.0);

    std::vector<Eigen::VectorXd> meas_list;
    for (const auto& tgt : targets) {
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            if (det_roll(rng) < p_detect) {
                int idx = timestep - tgt.birth_time;
                Eigen::VectorXd z(2);
                z(0) = tgt.states[idx](0) + noise(rng); // x + noise
                z(1) = tgt.states[idx](1) + noise(rng); // y + noise
                meas_list.push_back(z);
            }
        }
    }

    if (meas_list.empty()) {
        return Eigen::MatrixXd(2, 0);
    }

    Eigen::MatrixXd Z(2, static_cast<int>(meas_list.size()));
    for (int j = 0; j < static_cast<int>(meas_list.size()); ++j) {
        Z.col(j) = meas_list[j];
    }
    return Z;
}

// Find the closest extracted estimate to a truth position
static double closest_estimate_error(
    const distributions::Mixture<distributions::Gaussian>& estimates,
    const Eigen::VectorXd& truth_pos)
{
    double min_err = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        Eigen::VectorXd est_pos = estimates.component(i).mean().head(2);
        double err = (est_pos - truth_pos).norm();
        min_err = std::min(min_err, err);
    }
    return min_err;
}

// ---- Gaussian PHD tracking test ----

TEST(PHDTracking, TwoPointTargets) {
    std::mt19937 rng(42); // fixed seed for reproducibility

    const double dt = 1.0;
    const int num_steps = 30;
    const double meas_std = 1.0;
    const double p_detect = 0.95;

    // Dynamics: 2D constant velocity [x, y, vx, vy]
    auto dyn = std::make_shared<dynamics::Integrator2D>();

    // Two targets crossing paths
    Eigen::VectorXd x0_a(4), x0_b(4);
    x0_a << 0.0, 0.0, 2.0, 1.0;   // starts at origin, moves right+up
    x0_b << 50.0, 0.0, -1.0, 1.5;  // starts at (50,0), moves left+up

    auto target_a = make_linear_target(x0_a, 0, num_steps - 1, dt, *dyn);
    auto target_b = make_linear_target(x0_b, 0, num_steps - 1, dt, *dyn);

    std::vector<TruthTarget> targets = { target_a, target_b };

    // Setup EKF
    auto ekf = std::make_unique<filters::EKF>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(meas_std * meas_std * Eigen::MatrixXd::Identity(2, 2));

    // Birth model: two birth components covering the surveillance area
    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 100.0; birth_cov(1, 1) = 100.0;
    birth_cov(2, 2) = 10.0; birth_cov(3, 3) = 10.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.05);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.05);

    // Initial intensity (empty — will get birth on first predict)
    auto intensity = std::make_unique<distributions::Mixture<distributions::Gaussian>>();

    // PHD filter
    multi_target::PHD<distributions::Gaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(30);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);

#ifdef BREW_ENABLE_PLOTTING
    std::vector<double> truth_ax_vec, truth_ay_vec, truth_bx_vec, truth_by_vec;
    std::vector<double> meas_all_x, meas_all_y;
    std::vector<double> est_all_x, est_all_y;
#endif

    std::cout << "\n=== Gaussian PHD Tracking - Two Point Targets ===\n";
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err_A"
              << std::setw(12) << "Err_B"
              << "\n";
    std::cout << std::string(59, '-') << "\n";

    int converged_steps = 0;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);

        Eigen::MatrixXd meas = generate_measurements(targets, k, meas_std, rng, p_detect);

        if (meas.cols() > 0) {
            phd.correct(meas);
        } else {
            // No measurements: just do a correction with empty set
            phd.correct(Eigen::MatrixXd(2, 0));
        }

        phd.cleanup();

        // Get extracted estimates
        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        // Compute errors to truth
        Eigen::VectorXd truth_a = target_a.states[k].head(2);
        Eigen::VectorXd truth_b = target_b.states[k].head(2);

        double err_a = closest_estimate_error(latest, truth_a);
        double err_b = closest_estimate_error(latest, truth_b);

        std::cout << std::setw(5) << k
                  << std::setw(10) << meas.cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_a
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_b
                  << "\n";

        // After convergence period, check tracking accuracy
        if (k >= 10 && latest.size() >= 2) {
            if (err_a < 5.0 && err_b < 5.0) {
                converged_steps++;
            }
        }

#ifdef BREW_ENABLE_PLOTTING
        truth_ax_vec.push_back(truth_a(0));
        truth_ay_vec.push_back(truth_a(1));
        truth_bx_vec.push_back(truth_b(0));
        truth_by_vec.push_back(truth_b(1));
        for (int j = 0; j < meas.cols(); ++j) {
            meas_all_x.push_back(meas(0, j));
            meas_all_y.push_back(meas(1, j));
        }
        for (std::size_t i = 0; i < latest.size(); ++i) {
            est_all_x.push_back(latest.component(i).mean()(0));
            est_all_y.push_back(latest.component(i).mean()(1));
        }
#endif
    }

    std::cout << "\nConverged steps (k>=10, both errors < 5.0): "
              << converged_steps << " / " << (num_steps - 10) << "\n";

    // We expect most steps after convergence to track both targets
    EXPECT_GE(converged_steps, 10) << "PHD should track both targets for most of the run";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000);
        fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        // Measurements (light gray dots)
        if (!meas_all_x.empty()) {
            auto mp = ax->plot(meas_all_x, meas_all_y, ".");
            mp->color({0.f, 0.7f, 0.7f, 0.7f});
            mp->marker_size(4.0f);
        }

        // Truth trajectory A (solid black)
        auto ta = ax->plot(truth_ax_vec, truth_ay_vec);
        ta->color({0.f, 0.f, 0.f, 0.f});
        ta->line_width(2.5f);

        // Truth trajectory B (dashed black)
        auto tb = ax->plot(truth_bx_vec, truth_by_vec, "--");
        tb->color({0.f, 0.f, 0.f, 0.f});
        tb->line_width(2.5f);

        // Estimates (MATLAB blue dots)
        if (!est_all_x.empty()) {
            auto ep = ax->plot(est_all_x, est_all_y, ".");
            ep->color({0.f, 0.f, 0.4470f, 0.7410f});
            ep->marker_size(8.0f);
        }

        // Covariance ellipses at every timestep
        const auto& all_extracted = phd.extracted_mixtures();
        for (const auto& mix_ptr : all_extracted) {
            for (std::size_t i = 0; i < mix_ptr->size(); ++i) {
                brew::plot_utils::plot_gaussian_2d(ax, mix_ptr->component(i),
                    {0, 1}, {0.f, 0.f, 0.4470f, 0.7410f}, 2.0, 0.7f);
            }
        }

        ax->title("Gaussian PHD - Two Point Targets");
        ax->xlabel("x");
        ax->ylabel("y");

        brew::plot_utils::save_figure(fig, "/workspace/brew/output/phd_gaussian_two_targets.png");
    }
#endif
}

// ---- GGIW PHD tracking test (extended target) ----

TEST(PHDTracking, ExtendedTargetGGIW) {
    std::mt19937 rng(123);

    const double dt = 1.0;
    const int num_steps = 20;
    const double meas_std = 0.5;

    auto dyn = std::make_shared<dynamics::Integrator2D>();

    // Single extended target moving right
    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.0, 2.0, 0.5;
    auto target = make_linear_target(x0, 0, num_steps - 1, dt, *dyn);

    // Setup GGIW EKF
    auto ggiw_ekf = std::make_unique<filters::GGIWEKF>();
    ggiw_ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ggiw_ekf->set_measurement_jacobian(H);
    ggiw_ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ggiw_ekf->set_measurement_noise(meas_std * meas_std * Eigen::MatrixXd::Identity(2, 2));
    ggiw_ekf->set_temporal_decay(1.0);
    ggiw_ekf->set_forgetting_factor(5.0);
    ggiw_ekf->set_scaling_parameter(1.0);

    // Birth model
    auto birth = std::make_unique<distributions::Mixture<distributions::GGIW>>();
    Eigen::VectorXd b_mean(4);
    b_mean << 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd b_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 20.0 * Eigen::MatrixXd::Identity(2, 2);
    birth->add_component(
        std::make_unique<distributions::GGIW>(b_mean, b_cov, 10.0, 1.0, 10.0, b_V), 0.1);

    auto intensity = std::make_unique<distributions::Mixture<distributions::GGIW>>();

    multi_target::PHD<distributions::GGIW> phd;
    phd.set_filter(std::move(ggiw_ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(0.9);
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
    std::vector<double> truth_x_vec, truth_y_vec;
    std::vector<double> meas_all_x, meas_all_y;
    std::vector<double> est_all_x, est_all_y;
#endif

    std::cout << "\n=== GGIW PHD Tracking - Single Extended Target ===\n";
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err"
              << "\n";
    std::cout << std::string(47, '-') << "\n";

    std::normal_distribution<double> noise(0.0, meas_std);
    // Extent: measurements come from an ellipse around truth position
    Eigen::Matrix2d extent;
    extent << 4.0, 1.0, 1.0, 2.0;
    Eigen::LLT<Eigen::MatrixXd> llt(extent);
    Eigen::MatrixXd L_ext = llt.matrixL();

    int tracked_steps = 0;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);

        // Generate multiple measurements from the extended target
        int num_meas = 3 + (rng() % 5); // 3-7 measurements
        Eigen::MatrixXd meas(2, num_meas);
        Eigen::VectorXd truth_pos = target.states[k].head(2);

        for (int j = 0; j < num_meas; ++j) {
            Eigen::VectorXd z(2);
            z(0) = noise(rng);
            z(1) = noise(rng);
            meas.col(j) = truth_pos + L_ext * z;
        }

        // For extended target, we pass all measurements at once
        // The PHD will cluster them (no cluster object set, so each column treated separately)
        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        double err = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < latest.size(); ++i) {
            Eigen::VectorXd est_pos = latest.component(i).mean().head(2);
            err = std::min(err, (est_pos - truth_pos).norm());
        }

        std::cout << std::setw(5) << k
                  << std::setw(10) << num_meas
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << (std::isfinite(err) ? err : -1.0)
                  << "\n";

        if (k >= 5 && latest.size() >= 1 && err < 10.0) {
            tracked_steps++;
        }

#ifdef BREW_ENABLE_PLOTTING
        truth_x_vec.push_back(truth_pos(0));
        truth_y_vec.push_back(truth_pos(1));
        for (int j = 0; j < meas.cols(); ++j) {
            meas_all_x.push_back(meas(0, j));
            meas_all_y.push_back(meas(1, j));
        }
        for (std::size_t i = 0; i < latest.size(); ++i) {
            est_all_x.push_back(latest.component(i).mean()(0));
            est_all_y.push_back(latest.component(i).mean()(1));
        }
#endif
    }

    std::cout << "\nTracked steps (k>=5, error < 10.0): "
              << tracked_steps << " / " << (num_steps - 5) << "\n";

    EXPECT_GE(tracked_steps, 5) << "GGIW PHD should track the extended target";

#ifdef BREW_ENABLE_PLOTTING
    {
        std::filesystem::create_directories("/workspace/brew/output");
        auto fig = matplot::figure(true);
        fig->width(1000);
        fig->height(800);
        auto ax = fig->current_axes();
        ax->hold(true);

        // Measurements (light gray dots)
        if (!meas_all_x.empty()) {
            auto mp = ax->plot(meas_all_x, meas_all_y, ".");
            mp->color({0.f, 0.7f, 0.7f, 0.7f});
            mp->marker_size(4.0f);
        }

        // Truth trajectory (solid black)
        auto tl = ax->plot(truth_x_vec, truth_y_vec);
        tl->color({0.f, 0.f, 0.f, 0.f});
        tl->line_width(2.5f);

        // Estimates (MATLAB blue dots)
        if (!est_all_x.empty()) {
            auto ep = ax->plot(est_all_x, est_all_y, ".");
            ep->color({0.f, 0.f, 0.4470f, 0.7410f});
            ep->marker_size(8.0f);
        }

        // GGIW extent ellipses at every timestep
        const auto& all_extracted = phd.extracted_mixtures();
        for (const auto& mix_ptr : all_extracted) {
            for (std::size_t i = 0; i < mix_ptr->size(); ++i) {
                brew::plot_utils::plot_ggiw_2d(ax, mix_ptr->component(i),
                    {0, 1}, {0.f, 0.f, 0.4470f, 0.7410f});
            }
        }

        ax->title("GGIW PHD - Extended Target Tracking");
        ax->xlabel("x");
        ax->ylabel("y");

        brew::plot_utils::save_figure(fig, "/workspace/brew/output/phd_ggiw_extended_target.png");
    }
#endif
}
