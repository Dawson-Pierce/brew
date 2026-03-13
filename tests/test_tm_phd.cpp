#include <gtest/gtest.h>
#include "brew/dynamics/single_integrator.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/mixture.hpp"
#include "brew/filters/tm_ekf.hpp"
#include "brew/filters/trajectory_tm_ekf.hpp"
#include "brew/multi_target/phd.hpp"
#include "brew/fusion/prune.hpp"
#include "brew/fusion/merge.hpp"
#include "brew/clustering/dbscan.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include "brew/template_matching/point_cloud_io.hpp"
#include "brew/template_matching/point_to_point_icp.hpp"

#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/plot_utils/plot_options.hpp>
#include <brew/plot_utils/color_palette.hpp>
#include <brew/plot_utils/plot_point_cloud.hpp>
#include <filesystem>
#endif

using namespace brew;

namespace {

// Transform a 2D point cloud by rotation angle and translation
Eigen::MatrixXd transform_2d(const Eigen::MatrixXd& pts,
                              double angle,
                              const Eigen::Vector2d& translation) {
    Eigen::Matrix2d R;
    R << std::cos(angle), -std::sin(angle),
         std::sin(angle),  std::cos(angle);
    Eigen::MatrixXd out = R * pts;
    out.colwise() += translation;
    return out;
}

// Truth target with rotation
struct TmTruthTarget {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states;  // [x, y, vx, vy]
    std::vector<double> angles;            // rotation angle at each step
};

TmTruthTarget make_tm_target(
    const Eigen::VectorXd& x0, double angle0, double angular_vel,
    int birth, int death, double dt,
    dynamics::DynamicsBase& dyn)
{
    TmTruthTarget t;
    t.birth_time = birth;
    t.death_time = death;
    Eigen::VectorXd x = x0;
    double angle = angle0;
    for (int k = birth; k <= death; ++k) {
        t.states.push_back(x);
        t.angles.push_back(angle);
        x = dyn.propagate_state(dt, x);
        angle += angular_vel * dt;
    }
    return t;
}

// Generate point cloud measurements from truth targets
Eigen::MatrixXd generate_tm_measurements(
    const std::vector<TmTruthTarget>& targets,
    const template_matching::PointCloud& templ,
    int timestep, double point_noise_std,
    std::mt19937& rng, double p_detect)
{
    std::normal_distribution<double> noise(0.0, point_noise_std);
    std::uniform_real_distribution<double> det_roll(0.0, 1.0);

    std::vector<Eigen::MatrixXd> clouds;

    for (const auto& tgt : targets) {
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            if (det_roll(rng) < p_detect) {
                int idx = timestep - tgt.birth_time;
                Eigen::Vector2d pos = tgt.states[idx].head(2);
                double angle = tgt.angles[idx];

                Eigen::MatrixXd pts = transform_2d(templ.points(), angle, pos);
                for (int j = 0; j < pts.cols(); ++j) {
                    pts(0, j) += noise(rng);
                    pts(1, j) += noise(rng);
                }
                clouds.push_back(pts);
            }
        }
    }

    if (clouds.empty()) return Eigen::MatrixXd(2, 0);

    int total = 0;
    for (const auto& c : clouds) total += static_cast<int>(c.cols());
    Eigen::MatrixXd Z(2, total);
    int col = 0;
    for (const auto& c : clouds) {
        Z.block(0, col, 2, c.cols()) = c;
        col += static_cast<int>(c.cols());
    }
    return Z;
}

// Compute rotation error (angle difference in radians)
double rotation_error(const Eigen::MatrixXd& R_est, double truth_angle) {
    double est_angle = std::atan2(R_est(1, 0), R_est(0, 0));
    double err = est_angle - truth_angle;
    // Wrap to [-pi, pi]
    while (err > std::numbers::pi) err -= 2.0 * std::numbers::pi;
    while (err < -std::numbers::pi) err += 2.0 * std::numbers::pi;
    return std::abs(err);
}

// Create a TmEkf filter
std::unique_ptr<filters::TmEkf> make_tm_ekf(
    std::shared_ptr<dynamics::DynamicsBase> dyn,
    std::shared_ptr<template_matching::IcpBase> icp)
{
    auto ekf = std::make_unique<filters::TmEkf>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(3, 3);
    R_meas(0, 0) = 0.5;
    R_meas(1, 1) = 0.5;
    R_meas(2, 2) = 0.001;  // trust ICP rotation (no angular velocity in state)
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(1.0 * Eigen::MatrixXd::Identity(1, 1));
    ekf->set_icp(icp);
    return ekf;
}

// Create a TrajectoryTmEkf filter
std::unique_ptr<filters::TrajectoryTmEkf> make_trajectory_tm_ekf(
    std::shared_ptr<dynamics::DynamicsBase> dyn,
    std::shared_ptr<template_matching::IcpBase> icp,
    int window = 10)
{
    auto ekf = std::make_unique<filters::TrajectoryTmEkf>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(3, 3);
    R_meas(0, 0) = 0.5;
    R_meas(1, 1) = 0.5;
    R_meas(2, 2) = 0.001;  // trust ICP rotation (no angular velocity in state)
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(1.0 * Eigen::MatrixXd::Identity(1, 1));
    ekf->set_icp(icp);
    ekf->set_window_size(window);
    return ekf;
}

// Create TemplatePose birth model — one component per template, centered in scene
std::unique_ptr<models::Mixture<models::TemplatePose>> make_tm_birth(
    std::shared_ptr<template_matching::PointCloud> templ_rect,
    std::shared_ptr<template_matching::PointCloud> templ_tri,
    double weight = 0.05)
{
    auto birth = std::make_unique<models::Mixture<models::TemplatePose>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(5, 5);
    birth_cov(0, 0) = 2500.0;  // high location covariance — centered in middle
    birth_cov(1, 1) = 2500.0;
    birth_cov(2, 2) = 25.0;
    birth_cov(3, 3) = 25.0;
    birth_cov(4, 4) = 1.0;
    Eigen::Matrix2d R_init = Eigen::Matrix2d::Identity();
    std::vector<int> pos_indices = {0, 1};

    // Both births centered at approximate scene center
    Eigen::VectorXd b0(4);
    b0 << 25.0, 15.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TemplatePose>(b0, birth_cov, R_init, templ_rect, pos_indices), weight);
    birth->add_component(
        std::make_unique<models::TemplatePose>(b0, birth_cov, R_init, templ_tri, pos_indices), weight);
    return birth;
}

// Create Trajectory<TemplatePose> birth model — one component per template, centered in scene
std::unique_ptr<models::Mixture<models::Trajectory<models::TemplatePose>>>
make_trajectory_tm_birth(
    std::shared_ptr<template_matching::PointCloud> templ_rect,
    std::shared_ptr<template_matching::PointCloud> templ_tri,
    double weight = 0.05)
{
    auto birth = std::make_unique<models::Mixture<models::Trajectory<models::TemplatePose>>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(5, 5);
    birth_cov(0, 0) = 2500.0;  // high location covariance — centered in middle
    birth_cov(1, 1) = 2500.0;
    birth_cov(2, 2) = 25.0;
    birth_cov(3, 3) = 25.0;
    birth_cov(4, 4) = 1.0;
    Eigen::Matrix2d R_init = Eigen::Matrix2d::Identity();
    std::vector<int> pos_indices = {0, 1};

    Eigen::VectorXd b0(4);
    b0 << 25.0, 15.0, 0.0, 0.0;

    auto tp1 = models::TemplatePose(b0, birth_cov, R_init, templ_rect, pos_indices);
    auto tp2 = models::TemplatePose(b0, birth_cov, R_init, templ_tri, pos_indices);

    // Build Trajectory manually: stacked state = translational mean, stacked cov = translational cov
    auto traj1 = std::make_unique<models::Trajectory<models::TemplatePose>>();
    traj1->state_dim = tp1.trans_dim();
    traj1->mean() = tp1.mean();
    traj1->covariance() = tp1.covariance().topLeftCorner(tp1.trans_dim(), tp1.trans_dim());
    traj1->history().push_back(std::move(tp1));

    auto traj2 = std::make_unique<models::Trajectory<models::TemplatePose>>();
    traj2->state_dim = tp2.trans_dim();
    traj2->mean() = tp2.mean();
    traj2->covariance() = tp2.covariance().topLeftCorner(tp2.trans_dim(), tp2.trans_dim());
    traj2->history().push_back(std::move(tp2));

    birth->add_component(std::move(traj1), weight);
    birth->add_component(std::move(traj2), weight);
    return birth;
}

// Generate point cloud measurements with per-target templates
Eigen::MatrixXd generate_tm_measurements_multi(
    const std::vector<TmTruthTarget>& targets,
    const std::vector<std::shared_ptr<template_matching::PointCloud>>& templates,
    int timestep, double point_noise_std,
    std::mt19937& rng, double p_detect)
{
    std::normal_distribution<double> noise(0.0, point_noise_std);
    std::uniform_real_distribution<double> det_roll(0.0, 1.0);

    std::vector<Eigen::MatrixXd> clouds;

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& tgt = targets[ti];
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            if (det_roll(rng) < p_detect) {
                int idx = timestep - tgt.birth_time;
                Eigen::Vector2d pos = tgt.states[idx].head(2);
                double angle = tgt.angles[idx];

                Eigen::MatrixXd pts = transform_2d(templates[ti]->points(), angle, pos);
                for (int j = 0; j < pts.cols(); ++j) {
                    pts(0, j) += noise(rng);
                    pts(1, j) += noise(rng);
                }
                clouds.push_back(pts);
            }
        }
    }

    if (clouds.empty()) return Eigen::MatrixXd(2, 0);

    int total = 0;
    for (const auto& c : clouds) total += static_cast<int>(c.cols());
    Eigen::MatrixXd Z(2, total);
    int col = 0;
    for (const auto& c : clouds) {
        Z.block(0, col, 2, c.cols()) = c;
        col += static_cast<int>(c.cols());
    }
    return Z;
}

// Shared scenario setup
struct TmScenario {
    int num_steps = 30;
    double dt = 1.0;
    double point_noise_std = 0.05;  // low noise — this is an algorithm test
    double p_detect = 1.0;          // perfect detection for clean test

    std::shared_ptr<template_matching::PointCloud> templ_rect;  // rectangle
    std::shared_ptr<template_matching::PointCloud> templ_tri;     // L-shape
    std::vector<std::shared_ptr<template_matching::PointCloud>> target_templates;  // per-target
    std::shared_ptr<dynamics::SingleIntegrator> dyn;
    std::shared_ptr<template_matching::PointToPointIcp> icp;
    std::vector<TmTruthTarget> truth_targets;
    std::vector<Eigen::MatrixXd> measurements;

    void setup() {
        templ_rect = std::make_shared<template_matching::PointCloud>(
            template_matching::sample_rectangle(3.0, 1.5, 20).points());
        templ_tri = std::make_shared<template_matching::PointCloud>(
            template_matching::sample_triangle(3.0, 2.5, 20).points());
        dyn = std::make_shared<dynamics::SingleIntegrator>(2);

        icp = std::make_shared<template_matching::PointToPointIcp>();
        template_matching::IcpParams icp_params;
        icp_params.max_iterations = 50;
        icp_params.tolerance = 1e-6;
        icp_params.sigma_sq = 0.1;
        icp->set_params(icp_params);

        // Two targets: target A uses rectangle, target B uses L-shape
        Eigen::VectorXd x0_a(4), x0_b(4);
        x0_a << 0.0, 0.0, 2.0, 0.5;
        x0_b << 50.0, 0.0, -0.5, 1.5;

        truth_targets.push_back(
            make_tm_target(x0_a, 0.0, 0.1, 0, num_steps - 1, dt, *dyn));
        truth_targets.push_back(
            make_tm_target(x0_b, std::numbers::pi / 4.0, -0.08, 0, num_steps - 1, dt, *dyn));

        target_templates = {templ_rect, templ_tri};

        std::mt19937 rng(42);
        measurements.resize(num_steps);
        for (int k = 0; k < num_steps; ++k) {
            measurements[k] = generate_tm_measurements_multi(
                truth_targets, target_templates, k, point_noise_std, rng, p_detect);
        }
    }
};

#ifdef BREW_ENABLE_PLOTTING
// Plot transformed template at a given pose
void plot_template_at_pose(matplot::axes_handle ax,
                           const template_matching::PointCloud& templ,
                           double angle, const Eigen::Vector2d& pos,
                           const brew::plot_utils::Color& color,
                           float marker_size = 5.0f) {
    Eigen::MatrixXd pts = transform_2d(templ.points(), angle, pos);
    template_matching::PointCloud cloud(pts);
    brew::plot_utils::plot_point_cloud_2d(ax, cloud, {0, 1}, color, marker_size);
}
#endif

} // anonymous namespace


// ============================================================
// TM-PHD: shape + position tracking test
// ============================================================

TEST(TmPhd, Tracking2D) {
    TmScenario sc;
    sc.setup();

    auto tm_ekf = make_tm_ekf(sc.dyn, sc.icp);
    auto birth = make_tm_birth(sc.templ_rect, sc.templ_tri);

    multi_target::PHD<models::TemplatePose> phd;
    phd.set_filter(std::move(tm_ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::TemplatePose>>());
    phd.set_prob_detection(sc.p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(30);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(50.0);
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(8.0, 3));

    std::cout << "\n=== TM-PHD 2D Tracking ===\n";
    std::cout << std::setw(5) << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est"
              << std::setw(12) << "pos_a"
              << std::setw(12) << "pos_b"
              << std::setw(12) << "rot_a"
              << std::setw(12) << "rot_b"
              << "\n";

    int converged_steps = 0;
    const int warmup = 5;
    const double pos_thresh = 3.0;
    const double rot_thresh = 0.5;  // ~30 degrees

    for (int k = 0; k < sc.num_steps; ++k) {
        phd.predict(k, sc.dt);
        phd.correct(sc.measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();

        // For each truth target, find closest estimate
        double pos_a = std::numeric_limits<double>::infinity();
        double pos_b = std::numeric_limits<double>::infinity();
        double rot_a = std::numeric_limits<double>::infinity();
        double rot_b = std::numeric_limits<double>::infinity();

        for (int t = 0; t < 2; ++t) {
            const auto& tgt = sc.truth_targets[t];
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                double truth_angle = tgt.angles[idx];

                double best_pos = std::numeric_limits<double>::infinity();
                double best_rot = std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < latest.size(); ++i) {
                    double pe = (latest.component(i).mean().head(2) - truth_pos).norm();
                    if (pe < best_pos) {
                        best_pos = pe;
                        best_rot = rotation_error(latest.component(i).rotation(), truth_angle);
                    }
                }
                if (t == 0) { pos_a = best_pos; rot_a = best_rot; }
                else        { pos_b = best_pos; rot_b = best_rot; }
            }
        }

        bool good = (k >= warmup)
                   && pos_a < pos_thresh && pos_b < pos_thresh
                   && rot_a < rot_thresh && rot_b < rot_thresh;
        if (good) converged_steps++;

        std::cout << std::setw(5) << k
                  << std::setw(10) << sc.measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << pos_a
                  << std::setw(12) << pos_b
                  << std::setw(12) << std::setprecision(3) << rot_a
                  << std::setw(12) << rot_b
                  << "\n";
    }

    std::cout << "Converged steps (k>=" << warmup
              << ", pos<" << pos_thresh << ", rot<" << rot_thresh << "): "
              << converged_steps << " / " << (sc.num_steps - warmup) << "\n";

    EXPECT_GE(converged_steps, 15)
        << "TM-PHD should track both targets' position and rotation";

#ifdef BREW_ENABLE_PLOTTING
    std::filesystem::create_directories("output");
    const int plot_stride = 1;  // plot every Nth timestep

    // --- Figure 1: Main tracking plot (measurements gray, truth black, estimates color) ---
    {
        auto fig = matplot::figure(true);
        fig->width(900);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        // Measurements in gray
        for (int k = 0; k < sc.num_steps; k += plot_stride) {
            const auto& m = sc.measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols());
            for (int j = 0; j < m.cols(); ++j) { mx[j] = m(0, j); my[j] = m(1, j); }
            auto mp = ax->plot(mx, my, ".");
            mp->color({0.f, 0.7f, 0.7f, 0.7f});
            mp->marker_size(2.0f);
        }

        // Truth trajectories + templates in black
        for (int t = 0; t < 2; ++t) {
            std::vector<double> tx, ty;
            for (const auto& s : sc.truth_targets[t].states) {
                tx.push_back(s(0)); ty.push_back(s(1));
            }
            auto tl = ax->plot(tx, ty);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(1.0f);
        }
        for (int k = 0; k < sc.num_steps; k += plot_stride) {
            for (int t = 0; t < 2; ++t) {
                const auto& tgt = sc.truth_targets[t];
                if (k >= tgt.birth_time && k <= tgt.death_time) {
                    int idx = k - tgt.birth_time;
                    Eigen::Vector2d pos = tgt.states[idx].head(2);
                    plot_template_at_pose(ax, *sc.target_templates[t], tgt.angles[idx], pos,
                        {0.f, 0.f, 0.f, 0.f}, 3.0f);
                }
            }
        }

        // Estimated trajectory lines in color
        std::vector<std::vector<double>> ex(2), ey(2);
        for (int k = 0; k < sc.num_steps; ++k) {
            const auto& mix = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < mix.size(); ++i) {
                Eigen::Vector2d pos = mix.component(i).mean().head(2);
                int best_t = 0;
                double best_d = std::numeric_limits<double>::infinity();
                for (int t = 0; t < 2; ++t) {
                    if (k >= sc.truth_targets[t].birth_time && k <= sc.truth_targets[t].death_time) {
                        int idx = k - sc.truth_targets[t].birth_time;
                        double d = (pos - sc.truth_targets[t].states[idx].head(2)).norm();
                        if (d < best_d) { best_d = d; best_t = t; }
                    }
                }
                if (best_d < 10.0) {
                    ex[best_t].push_back(pos(0));
                    ey[best_t].push_back(pos(1));
                }
            }
        }
        for (int t = 0; t < 2; ++t) {
            if (ex[t].size() < 2) continue;
            auto c = brew::plot_utils::lines_color(t);
            auto el = ax->plot(ex[t], ey[t]);
            el->color({c[0], c[1], c[2], c[3]});
            el->line_width(1.5f);
        }

        // Estimated templates in color
        for (int k = 0; k < sc.num_steps; k += plot_stride) {
            const auto& mix = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < mix.size(); ++i) {
                const auto& comp = mix.component(i);
                Eigen::Vector2d pos = comp.mean().head(2);
                double angle = std::atan2(comp.rotation()(1, 0), comp.rotation()(0, 0));
                auto c = brew::plot_utils::lines_color(static_cast<int>(i));
                plot_template_at_pose(ax, comp.get_template(), angle, pos, c, 3.0f);
            }
        }

        // ax->title("TM-PHD Tracking");
        ax->xlabel("X"); ax->ylabel("Y");
        ax->x_axis().label_font_size(18);
        ax->y_axis().label_font_size(18);
        ax->axes_aspect_ratio_auto(false);
        ax->axes_aspect_ratio(1.0f);
        brew::plot_utils::save_figure(fig, "output/tm_phd_2d.png");
    }

    // --- Figure 2: Convergence (position error + rotation error over time) ---
    {
        auto fig = matplot::figure(true);
        fig->width(900);
        fig->height(500);

        // Collect per-step errors
        std::vector<double> steps, pa, pb, ra, rb;
        for (int k = 0; k < sc.num_steps; ++k) {
            steps.push_back(static_cast<double>(k));
            const auto& mix = *phd.extracted_mixtures()[k];
            double best_pos_a = std::numeric_limits<double>::infinity();
            double best_rot_a = std::numeric_limits<double>::infinity();
            double best_pos_b = std::numeric_limits<double>::infinity();
            double best_rot_b = std::numeric_limits<double>::infinity();
            for (int t = 0; t < 2; ++t) {
                const auto& tgt = sc.truth_targets[t];
                if (k >= tgt.birth_time && k <= tgt.death_time) {
                    int idx = k - tgt.birth_time;
                    Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                    double truth_angle = tgt.angles[idx];
                    double bp = std::numeric_limits<double>::infinity();
                    double br = std::numeric_limits<double>::infinity();
                    for (std::size_t i = 0; i < mix.size(); ++i) {
                        double pe = (mix.component(i).mean().head(2) - truth_pos).norm();
                        if (pe < bp) {
                            bp = pe;
                            br = rotation_error(mix.component(i).rotation(), truth_angle);
                        }
                    }
                    if (t == 0) { best_pos_a = bp; best_rot_a = br; }
                    else        { best_pos_b = bp; best_rot_b = br; }
                }
            }
            pa.push_back(best_pos_a); pb.push_back(best_pos_b);
            ra.push_back(best_rot_a); rb.push_back(best_rot_b);
        }

        auto ax1 = matplot::subplot(fig, 1, 2, 0);
        ax1->hold(true);
        auto la = ax1->plot(steps, pa); la->display_name("Target A");
        auto lb = ax1->plot(steps, pb); lb->display_name("Target B");
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra_l = ax2->plot(steps, ra); ra_l->display_name("Target A");
        auto rb_l = ax2->plot(steps, rb); rb_l->display_name("Target B");
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/tm_phd_2d_convergence.png");
    }
#endif
}


// ============================================================
// Trajectory TM-PHD: trajectory + shape tracking test
// ============================================================

TEST(TmPhd, TrajectoryTracking2D) {
    TmScenario sc;
    sc.setup();

    auto tm_ekf = make_trajectory_tm_ekf(sc.dyn, sc.icp, 10);
    auto birth = make_trajectory_tm_birth(sc.templ_rect, sc.templ_tri);

    multi_target::PHD<models::Trajectory<models::TemplatePose>> phd;
    phd.set_filter(std::move(tm_ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(
        std::make_unique<models::Mixture<models::Trajectory<models::TemplatePose>>>());
    phd.set_prob_detection(sc.p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(30);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(50.0);
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(8.0, 3));

    std::cout << "\n=== Trajectory TM-PHD 2D Tracking ===\n";
    std::cout << std::setw(5) << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est"
              << std::setw(12) << "pos_a"
              << std::setw(12) << "pos_b"
              << std::setw(12) << "rot_a"
              << std::setw(12) << "rot_b"
              << "\n";

    int converged_steps = 0;
    const int warmup = 5;
    const double pos_thresh = 3.0;
    const double rot_thresh = 0.5;

    for (int k = 0; k < sc.num_steps; ++k) {
        phd.predict(k, sc.dt);
        phd.correct(sc.measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();

        double pos_a = std::numeric_limits<double>::infinity();
        double pos_b = std::numeric_limits<double>::infinity();
        double rot_a = std::numeric_limits<double>::infinity();
        double rot_b = std::numeric_limits<double>::infinity();

        for (int t = 0; t < 2; ++t) {
            const auto& tgt = sc.truth_targets[t];
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                double truth_angle = tgt.angles[idx];

                double best_pos = std::numeric_limits<double>::infinity();
                double best_rot = std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < latest.size(); ++i) {
                    // For Trajectory<TemplatePose>, current() gives the latest TemplatePose
                    const auto& tp = latest.component(i).current();
                    double pe = (tp.mean().head(2) - truth_pos).norm();
                    if (pe < best_pos) {
                        best_pos = pe;
                        best_rot = rotation_error(tp.rotation(), truth_angle);
                    }
                }
                if (t == 0) { pos_a = best_pos; rot_a = best_rot; }
                else        { pos_b = best_pos; rot_b = best_rot; }
            }
        }

        bool good = (k >= warmup)
                   && pos_a < pos_thresh && pos_b < pos_thresh
                   && rot_a < rot_thresh && rot_b < rot_thresh;
        if (good) converged_steps++;

        std::cout << std::setw(5) << k
                  << std::setw(10) << sc.measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(2) << pos_a
                  << std::setw(12) << pos_b
                  << std::setw(12) << std::setprecision(3) << rot_a
                  << std::setw(12) << rot_b
                  << "\n";
    }

    std::cout << "Converged steps (k>=" << warmup
              << ", pos<" << pos_thresh << ", rot<" << rot_thresh << "): "
              << converged_steps << " / " << (sc.num_steps - warmup) << "\n";

    EXPECT_GE(converged_steps, 15)
        << "Trajectory TM-PHD should track both targets' position and rotation";

#ifdef BREW_ENABLE_PLOTTING
    std::filesystem::create_directories("output");
    const int plot_stride = 1;  // plot every Nth timestep

    // --- Figure 1: Main tracking plot (measurements gray, truth black, estimates color) ---
    {
        auto fig = matplot::figure(true);
        fig->width(900);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        // Measurements in gray
        for (int k = 0; k < sc.num_steps; k += plot_stride) {
            const auto& m = sc.measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols());
            for (int j = 0; j < m.cols(); ++j) { mx[j] = m(0, j); my[j] = m(1, j); }
            auto mp = ax->plot(mx, my, ".");
            mp->color({0.f, 0.7f, 0.7f, 0.7f});
            mp->marker_size(2.0f);
        }

        // Truth trajectories + templates in black
        for (int t = 0; t < 2; ++t) {
            std::vector<double> tx, ty;
            for (const auto& s : sc.truth_targets[t].states) {
                tx.push_back(s(0)); ty.push_back(s(1));
            }
            auto tl = ax->plot(tx, ty);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(1.0f);
        }
        for (int k = 0; k < sc.num_steps; k += plot_stride) {
            for (int t = 0; t < 2; ++t) {
                const auto& tgt = sc.truth_targets[t];
                if (k >= tgt.birth_time && k <= tgt.death_time) {
                    int idx = k - tgt.birth_time;
                    Eigen::Vector2d pos = tgt.states[idx].head(2);
                    plot_template_at_pose(ax, *sc.target_templates[t], tgt.angles[idx], pos,
                        {0.f, 0.f, 0.f, 0.f}, 3.0f);
                }
            }
        }

        // Estimated trajectory lines in color (from final extraction's history)
        const auto& final_mix = *phd.extracted_mixtures().back();
        for (std::size_t i = 0; i < final_mix.size(); ++i) {
            const auto& traj = final_mix.component(i);
            const auto& hist = traj.history();
            if (hist.size() < 2) continue;
            std::vector<double> ex, ey;
            for (const auto& tp : hist) {
                ex.push_back(tp.mean()(0));
                ey.push_back(tp.mean()(1));
            }
            auto c = brew::plot_utils::lines_color(static_cast<int>(i));
            auto el = ax->plot(ex, ey);
            el->color({c[0], c[1], c[2], c[3]});
            el->line_width(1.5f);
        }

        // Estimated templates in color
        for (int k = 0; k < sc.num_steps; k += plot_stride) {
            const auto& mix = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < mix.size(); ++i) {
                const auto& tp = mix.component(i).current();
                Eigen::Vector2d pos = tp.mean().head(2);
                double angle = std::atan2(tp.rotation()(1, 0), tp.rotation()(0, 0));
                auto c = brew::plot_utils::lines_color(static_cast<int>(i));
                plot_template_at_pose(ax, tp.get_template(), angle, pos, c, 3.0f);
            }
        }

        // ax->title("Trajectory TM-PHD Tracking");
        ax->xlabel("X"); ax->ylabel("Y");
        ax->x_axis().label_font_size(18);
        ax->y_axis().label_font_size(18);
        ax->axes_aspect_ratio_auto(false);
        ax->axes_aspect_ratio(1.0f);
        brew::plot_utils::save_figure(fig, "output/trajectory_tm_phd_2d.png");
    }

    // --- Figure 2: Convergence (position error + rotation error over time) ---
    {
        auto fig = matplot::figure(true);
        fig->width(900);
        fig->height(500);

        std::vector<double> steps, pa, pb, ra, rb;
        for (int k = 0; k < sc.num_steps; ++k) {
            steps.push_back(static_cast<double>(k));
            const auto& mix = *phd.extracted_mixtures()[k];
            double best_pos_a = std::numeric_limits<double>::infinity();
            double best_rot_a = std::numeric_limits<double>::infinity();
            double best_pos_b = std::numeric_limits<double>::infinity();
            double best_rot_b = std::numeric_limits<double>::infinity();
            for (int t = 0; t < 2; ++t) {
                const auto& tgt = sc.truth_targets[t];
                if (k >= tgt.birth_time && k <= tgt.death_time) {
                    int idx = k - tgt.birth_time;
                    Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                    double truth_angle = tgt.angles[idx];
                    double bp = std::numeric_limits<double>::infinity();
                    double br = std::numeric_limits<double>::infinity();
                    for (std::size_t i = 0; i < mix.size(); ++i) {
                        const auto& tp = mix.component(i).current();
                        double pe = (tp.mean().head(2) - truth_pos).norm();
                        if (pe < bp) {
                            bp = pe;
                            br = rotation_error(tp.rotation(), truth_angle);
                        }
                    }
                    if (t == 0) { best_pos_a = bp; best_rot_a = br; }
                    else        { best_pos_b = bp; best_rot_b = br; }
                }
            }
            pa.push_back(best_pos_a); pb.push_back(best_pos_b);
            ra.push_back(best_rot_a); rb.push_back(best_rot_b);
        }

        auto ax1 = matplot::subplot(fig, 1, 2, 0);
        ax1->hold(true);
        auto la = ax1->plot(steps, pa); la->display_name("Target A");
        auto lb = ax1->plot(steps, pb); lb->display_name("Target B");
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra_l = ax2->plot(steps, ra); ra_l->display_name("Target A");
        auto rb_l = ax2->plot(steps, rb); rb_l->display_name("Target B");
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/trajectory_tm_phd_2d_convergence.png");
    }
#endif
}
