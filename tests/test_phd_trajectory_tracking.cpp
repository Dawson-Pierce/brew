#include <gtest/gtest.h>
#include "brew/multi_target/phd.hpp"
#include "brew/filters/trajectory_gaussian_ekf.hpp"
#include "brew/filters/trajectory_ggiw_ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/plot_utils/plot_options.hpp>
#include <brew/plot_utils/plot_trajectory_gaussian.hpp>
#include <brew/plot_utils/plot_trajectory_ggiw.hpp>
#include <brew/plot_utils/color_palette.hpp>
#include <filesystem>
#endif

using namespace brew;

// ---- Helpers ----

struct TrajectoryTruthTarget {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states;
};

static TrajectoryTruthTarget make_truth_target(
    const Eigen::VectorXd& initial_state,
    int birth_time, int death_time, double dt,
    dynamics::DynamicsBase& dyn)
{
    TrajectoryTruthTarget t;
    t.birth_time = birth_time;
    t.death_time = death_time;
    Eigen::VectorXd x = initial_state;
    for (int k = birth_time; k <= death_time; ++k) {
        t.states.push_back(x);
        x = dyn.propagate_state(dt, x);
    }
    return t;
}

static Eigen::MatrixXd generate_measurements(
    const std::vector<TrajectoryTruthTarget>& targets,
    int timestep, double meas_std,
    std::mt19937& rng, double p_detect)
{
    std::normal_distribution<double> noise(0.0, meas_std);
    std::uniform_real_distribution<double> det_roll(0.0, 1.0);

    std::vector<Eigen::VectorXd> meas_list;
    for (const auto& tgt : targets) {
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            if (det_roll(rng) < p_detect) {
                int idx = timestep - tgt.birth_time;
                Eigen::VectorXd z(2);
                z(0) = tgt.states[idx](0) + noise(rng);
                z(1) = tgt.states[idx](1) + noise(rng);
                meas_list.push_back(z);
            }
        }
    }

    if (meas_list.empty()) return Eigen::MatrixXd(2, 0);

    Eigen::MatrixXd Z(2, static_cast<int>(meas_list.size()));
    for (int j = 0; j < static_cast<int>(meas_list.size()); ++j) {
        Z.col(j) = meas_list[j];
    }
    return Z;
}

template <typename T>
static double closest_trajectory_error(
    const distributions::Mixture<T>& estimates,
    const Eigen::VectorXd& truth_pos)
{
    double min_err = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        Eigen::VectorXd est_pos = estimates.component(i).get_last_state().head(2);
        double err = (est_pos - truth_pos).norm();
        min_err = std::min(min_err, err);
    }
    return min_err;
}

// ---- Trajectory Gaussian PHD tracking test ----

TEST(PHDTrajectoryTracking, TwoPointTargets) {
    std::mt19937 rng(42);

    const double dt = 1.0;
    const int num_steps = 30;
    const double meas_std = 1.0;
    const double p_detect = 0.95;
    const int l_window = 10;

    auto dyn = std::make_shared<dynamics::Integrator2D>();

    // Two targets crossing paths
    Eigen::VectorXd x0_a(4), x0_b(4);
    x0_a << 0.0, 0.0, 2.0, 1.0;
    x0_b << 50.0, 0.0, -1.0, 1.5;

    auto target_a = make_truth_target(x0_a, 0, num_steps - 1, dt, *dyn);
    auto target_b = make_truth_target(x0_b, 0, num_steps - 1, dt, *dyn);
    std::vector<TrajectoryTruthTarget> targets = {target_a, target_b};

    // Trajectory Gaussian EKF
    auto ekf = std::make_unique<filters::TrajectoryGaussianEKF>();
    ekf->set_dynamics(dyn);
    ekf->set_window_size(l_window);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(meas_std * meas_std * Eigen::MatrixXd::Identity(2, 2));

    // Birth model: trajectory components at expected start positions
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

    std::cout << "\n=== Trajectory Gaussian PHD - Two Point Targets ===\n";
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
            phd.correct(Eigen::MatrixXd(2, 0));
        }

        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        Eigen::VectorXd truth_a = target_a.states[k].head(2);
        Eigen::VectorXd truth_b = target_b.states[k].head(2);

        double err_a = closest_trajectory_error(latest, truth_a);
        double err_b = closest_trajectory_error(latest, truth_b);

        std::cout << std::setw(5) << k
                  << std::setw(10) << meas.cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_a
                  << std::setw(12) << std::fixed << std::setprecision(2) << err_b
                  << "\n";

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
            auto pos = latest.component(i).get_last_state().head(2);
            est_all_x.push_back(pos(0));
            est_all_y.push_back(pos(1));
        }
#endif
    }

    std::cout << "\nConverged steps (k>=10, both errors < 5.0): "
              << converged_steps << " / " << (num_steps - 10) << "\n";

    EXPECT_GE(converged_steps, 10)
        << "Trajectory Gaussian PHD should track both targets";

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
        auto ta = ax->plot(truth_ax_vec, truth_ay_vec, "--");
        ta->color({0.f, 0.f, 0.f, 0.f});
        ta->line_width(2.5f);

        // Truth trajectory B (dashed black)
        auto tb = ax->plot(truth_bx_vec, truth_by_vec, "--");
        tb->color({0.f, 0.f, 0.f, 0.f});
        tb->line_width(2.5f);

        // Per-step estimates (faint blue dots)
        if (!est_all_x.empty()) {
            auto ep = ax->plot(est_all_x, est_all_y, ".");
            ep->color({0.f, 0.f, 0.4470f, 0.7410f});
            ep->marker_size(6.0f);
        }

        // Final extracted trajectory estimates (colored lines with markers)
        const auto& final_mix = *phd.extracted_mixtures().back();
        for (std::size_t i = 0; i < final_mix.size(); ++i) {
            auto c = brew::plot_utils::lines_color(static_cast<int>(i));
            brew::plot_utils::plot_trajectory_gaussian_2d(
                ax, final_mix.component(i), {0, 1}, c);
        }

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
    std::mt19937 rng(123);

    const double dt = 1.0;
    const int num_steps = 20;
    const double meas_std = 0.5;
    const int l_window = 10;

    auto dyn = std::make_shared<dynamics::Integrator2D>();

    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.0, 2.0, 0.5;
    auto target = make_truth_target(x0, 0, num_steps - 1, dt, *dyn);

    // Trajectory GGIW EKF
    auto ggiw_ekf = std::make_unique<filters::TrajectoryGGIWEKF>();
    ggiw_ekf->set_dynamics(dyn);
    ggiw_ekf->set_window_size(l_window);
    ggiw_ekf->set_temporal_decay(1.0);
    ggiw_ekf->set_forgetting_factor(5.0);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ggiw_ekf->set_measurement_jacobian(H);
    ggiw_ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ggiw_ekf->set_measurement_noise(meas_std * meas_std * Eigen::MatrixXd::Identity(2, 2));

    // Birth model
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

    std::cout << "\n=== Trajectory GGIW PHD - Extended Target ===\n";
    std::cout << std::setw(5) << "Step"
              << std::setw(10) << "N_meas"
              << std::setw(10) << "N_comp"
              << std::setw(10) << "N_est"
              << std::setw(12) << "Err"
              << "\n";
    std::cout << std::string(47, '-') << "\n";

    std::normal_distribution<double> noise(0.0, meas_std);
    Eigen::Matrix2d extent;
    extent << 4.0, 1.0, 1.0, 2.0;
    Eigen::LLT<Eigen::MatrixXd> llt(extent);
    Eigen::MatrixXd L_ext = llt.matrixL();

    int tracked_steps = 0;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);

        int num_meas = 3 + (rng() % 5);
        Eigen::MatrixXd meas(2, num_meas);
        Eigen::VectorXd truth_pos = target.states[k].head(2);

        for (int j = 0; j < num_meas; ++j) {
            Eigen::VectorXd z(2);
            z(0) = noise(rng);
            z(1) = noise(rng);
            meas.col(j) = truth_pos + L_ext * z;
        }

        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        double err = closest_trajectory_error(latest, truth_pos);

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
            auto pos = latest.component(i).get_last_state().head(2);
            est_all_x.push_back(pos(0));
            est_all_y.push_back(pos(1));
        }
#endif
    }

    std::cout << "\nTracked steps (k>=5, error < 10.0): "
              << tracked_steps << " / " << (num_steps - 5) << "\n";

    EXPECT_GE(tracked_steps, 5)
        << "Trajectory GGIW PHD should track the extended target";

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

        // Per-step estimates (faint blue dots)
        if (!est_all_x.empty()) {
            auto ep = ax->plot(est_all_x, est_all_y, ".");
            ep->color({0.f, 0.f, 0.4470f, 0.7410f});
            ep->marker_size(6.0f);
        }

        // Final extracted trajectory estimates with GGIW extents
        const auto& final_mix = *phd.extracted_mixtures().back();
        for (std::size_t i = 0; i < final_mix.size(); ++i) {
            auto c = brew::plot_utils::lines_color(static_cast<int>(i));
            brew::plot_utils::plot_trajectory_ggiw_2d(
                ax, final_mix.component(i), {0, 1}, c);
        }

        ax->title("Trajectory GGIW PHD - Extended Target");
        ax->xlabel("x");
        ax->ylabel("y");

        brew::plot_utils::save_figure(fig,
            "/workspace/brew/output/phd_traj_ggiw_extended_target.png");
    }
#endif
}
