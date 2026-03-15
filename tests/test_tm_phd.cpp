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
#include "brew/template_matching/point_to_plane_icp.hpp"
#include "brew/template_matching/pca_icp.hpp"

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
              << "  | weights"
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
                  << "  |";
        for (std::size_t i = 0; i < phd.intensity().size(); ++i) {
            std::cout << " " << std::setprecision(4) << phd.intensity().weight(i);
        }
        std::cout << "\n";
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
              << "  | weights"
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
                  << "  |";
        for (std::size_t i = 0; i < phd.intensity().size(); ++i) {
            std::cout << " " << std::setprecision(4) << phd.intensity().weight(i);
        }
        std::cout << "\n";
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


// ============================================================
// TM-PHD 3D: Plane STL template tracking test
// ============================================================

namespace {

// 3D rotation from axis-angle
Eigen::Matrix3d axis_angle_to_rotation(const Eigen::Vector3d& axis, double angle) {
    Eigen::Vector3d a = axis.normalized();
    Eigen::Matrix3d K;
    K <<     0.0, -a(2),  a(1),
          a(2),     0.0, -a(0),
         -a(1),  a(0),     0.0;
    return Eigen::Matrix3d::Identity()
           + std::sin(angle) * K
           + (1.0 - std::cos(angle)) * K * K;
}

// Transform a 3D point cloud by rotation matrix and translation
Eigen::MatrixXd transform_3d(const Eigen::MatrixXd& pts,
                              const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& translation) {
    Eigen::MatrixXd out = R * pts;
    out.colwise() += translation;
    return out;
}

// Truth target in 3D with rotation matrix
struct TmTruthTarget3D {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states;     // [x, y, z, vx, vy, vz]
    std::vector<Eigen::Matrix3d> rotations;  // rotation at each step
};

TmTruthTarget3D make_tm_target_3d(
    const Eigen::VectorXd& x0,
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& angular_vel,  // axis-angle rate per dt
    int birth, int death, double dt,
    dynamics::DynamicsBase& dyn)
{
    TmTruthTarget3D t;
    t.birth_time = birth;
    t.death_time = death;
    Eigen::VectorXd x = x0;
    Eigen::Matrix3d R = R0;
    for (int k = birth; k <= death; ++k) {
        t.states.push_back(x);
        t.rotations.push_back(R);
        x = dyn.propagate_state(dt, x);
        // Incremental rotation: R_{k+1} = R_k * Exp(omega * dt)
        Eigen::Vector3d phi = angular_vel * dt;
        double angle = phi.norm();
        if (angle > 1e-10) {
            R = R * axis_angle_to_rotation(phi, angle);
        }
    }
    return t;
}

// 3D rotation error: angle of R_err = R_est^T * R_truth
double rotation_error_3d(const Eigen::Matrix3d& R_est, const Eigen::Matrix3d& R_truth) {
    Eigen::Matrix3d R_err = R_est.transpose() * R_truth;
    double cos_angle = (R_err.trace() - 1.0) / 2.0;
    cos_angle = std::clamp(cos_angle, -1.0, 1.0);
    return std::acos(cos_angle);
}

// Generate 3D point cloud measurements from truth targets.
// Only points visible from below (face normal z < 0 after rotation) are included.
// normals: per-target normals matrices (3×N, matching templates).
// num_meas_pts: subsample visible points to this many (0 = use all visible).
Eigen::MatrixXd generate_tm_measurements_3d(
    const std::vector<TmTruthTarget3D>& targets,
    const std::vector<std::shared_ptr<template_matching::PointCloud>>& templates,
    const std::vector<Eigen::MatrixXd>& normals,
    int timestep, double point_noise_std,
    std::mt19937& rng, double p_detect,
    int num_meas_pts = 0)
{
    std::normal_distribution<double> noise(0.0, point_noise_std);
    std::uniform_real_distribution<double> det_roll(0.0, 1.0);
    std::vector<Eigen::MatrixXd> clouds;

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& tgt = targets[ti];
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            if (det_roll(rng) < p_detect) {
                int idx = timestep - tgt.birth_time;
                Eigen::Vector3d pos = tgt.states[idx].head(3);
                const Eigen::Matrix3d& R = tgt.rotations[idx];

                const Eigen::MatrixXd& src_pts = templates[ti]->points();
                const Eigen::MatrixXd& src_normals = normals[ti];

                // Filter to visible points (face normal z < 0 after rotation = visible from below)
                std::vector<int> visible;
                visible.reserve(src_pts.cols());
                for (int j = 0; j < src_pts.cols(); ++j) {
                    Eigen::Vector3d n_rot = R * src_normals.col(j);
                    if (n_rot(2) < 0.0) {
                        visible.push_back(j);
                    }
                }

                if (visible.empty()) continue;

                // Subsample visible points if requested
                if (num_meas_pts > 0 && num_meas_pts < static_cast<int>(visible.size())) {
                    std::shuffle(visible.begin(), visible.end(), rng);
                    visible.resize(num_meas_pts);
                }

                Eigen::MatrixXd pts(3, static_cast<int>(visible.size()));
                for (int j = 0; j < static_cast<int>(visible.size()); ++j) {
                    pts.col(j) = R * src_pts.col(visible[j]) + pos;
                    pts(0, j) += noise(rng);
                    pts(1, j) += noise(rng);
                    pts(2, j) += noise(rng);
                }
                clouds.push_back(pts);
            }
        }
    }

    if (clouds.empty()) return Eigen::MatrixXd(3, 0);

    int total = 0;
    for (const auto& c : clouds) total += static_cast<int>(c.cols());
    Eigen::MatrixXd Z(3, total);
    int col = 0;
    for (const auto& c : clouds) {
        Z.block(0, col, 3, c.cols()) = c;
        col += static_cast<int>(c.cols());
    }
    return Z;
}

} // anonymous namespace


TEST(TmPhd, Tracking3D_PlaneSTL) {
    // Surface-sample the STL mesh for the template (uniform coverage, not raw vertices)
    auto stl_cloud = template_matching::sample_stl("Plane.stl", 500);
    std::cout << "\nSampled Plane.stl template: " << stl_cloud.num_points() << " surface points, dim=" << stl_cloud.dim() << "\n";

    // Center using the surface-sampled centroid (must match measurement distribution)
    Eigen::Vector3d centroid = stl_cloud.points().rowwise().mean();
    stl_cloud.points().colwise() -= centroid;

    auto templ = std::make_shared<template_matching::PointCloud>(stl_cloud.points());
    std::cout << "Template: " << templ->num_points() << " points\n";
    std::cout << "Template bbox: X[" << templ->points().row(0).minCoeff() << ", " << templ->points().row(0).maxCoeff()
              << "] Y[" << templ->points().row(1).minCoeff() << ", " << templ->points().row(1).maxCoeff()
              << "] Z[" << templ->points().row(2).minCoeff() << ", " << templ->points().row(2).maxCoeff() << "]\n";

    // Dynamics: 3D single integrator, state = [x,y,z,vx,vy,vz]
    auto dyn = std::make_shared<dynamics::SingleIntegrator>(3);

    // PCA-aligned point-to-plane ICP (robust to arbitrary initial rotation)
    auto inner_icp = std::make_unique<template_matching::PointToPlaneIcp>();
    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 10;
    icp_params.tolerance = 1e-6;
    icp_params.sigma_sq = 1.0;
    inner_icp->set_params(icp_params);
    auto icp = std::make_shared<template_matching::PcaIcp>(std::move(inner_icp), *templ);

    // TM-EKF for 3D: state dim=6, rot_dim=3, aug_dim=9
    auto ekf = std::make_unique<filters::TmEkf>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(0.01 * Eigen::MatrixXd::Identity(3, 3));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.01 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(icp);
    auto* ekf_ptr = ekf.get();  // keep raw pointer for accessing ICP iterations

    // Scenario
    const int num_steps = 30;
    const double dt = 1.0;
    const double point_noise_std = 0.02;
    const double p_detect = 1.0;

    // Single target
    Eigen::VectorXd x0(6);
    x0 << 0.0, 0.0, 0.0, 5.0, 3.0, 1.0;

    // Initial rotation: align template's X-axis (fuselage) with velocity direction
    Eigen::Vector3d vel_dir = x0.tail(3).normalized();
    Eigen::Vector3d x_axis = -Eigen::Vector3d::UnitX();  // nose is along -X in template frame
    Eigen::Vector3d rot_axis = x_axis.cross(vel_dir);
    double rot_angle = std::acos(std::clamp(x_axis.dot(vel_dir), -1.0, 1.0));
    Eigen::Matrix3d R0 = (rot_axis.norm() > 1e-10)
        ? axis_angle_to_rotation(rot_axis.normalized(), rot_angle)
        : Eigen::Matrix3d::Identity();
    Eigen::Vector3d omega(0.0, 0.0, 0.05);  // slow yaw rotation

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0, R0, omega, 0, num_steps - 1, dt, *dyn));

    // Surface-sampled cloud with normals for visibility filtering
    Eigen::MatrixXd meas_normals;
    auto meas_cloud = template_matching::sample_stl("Plane.stl", 1000, meas_normals);
    meas_cloud.points().colwise() -= centroid;  // same centering as template
    auto meas_templ = std::make_shared<template_matching::PointCloud>(meas_cloud.points());
    std::cout << "Measurement source: " << meas_templ->num_points() << " surface-sampled points\n";

    std::vector<std::shared_ptr<template_matching::PointCloud>> target_templates = {meas_templ};
    std::vector<Eigen::MatrixXd> target_normals = {meas_normals};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, target_templates, target_normals,
            k, point_noise_std, rng, p_detect, 100);
    }

    // Birth model
    auto birth = std::make_unique<models::Mixture<models::TemplatePose>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 100.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 100.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = 1.0 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_init = Eigen::Matrix3d::Identity();  // PCA handles arbitrary init
    std::vector<int> pos_indices = {0, 1, 2};

    Eigen::VectorXd b0(6);
    b0 << 0.0, 0.0, 0.0, 5.0, 3.0, 1.0;
    birth->add_component(
        std::make_unique<models::TemplatePose>(b0, birth_cov, R_init, templ, pos_indices), 0.1);

    // PHD filter
    multi_target::PHD<models::TemplatePose> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::TemplatePose>>());
    phd.set_prob_detection(p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-6);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(30);
    phd.set_extract_threshold(0.3);
    phd.set_gate_threshold(100.0);
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(20.0, 3));

    std::cout << "\n=== TM-PHD 3D Tracking (Plane STL, Point-to-Plane ICP) ===\n";
    std::cout << std::setw(5) << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est"
              << std::setw(12) << "pos_err"
              << std::setw(12) << "rot_err"
              << std::setw(10) << "icp_iter"
              << "  | weights"
              << "\n";

    int converged_steps = 0;
    const int warmup = 5;
    const double pos_thresh = 3.0;
    const double rot_thresh = 0.5;

    // Store errors for convergence plot
    std::vector<double> pos_errors, rot_errors;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);
        phd.correct(measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();

        const auto& tgt = truth_targets[0];
        int idx = k - tgt.birth_time;
        Eigen::Vector3d truth_pos = tgt.states[idx].head(3);
        const Eigen::Matrix3d& truth_R = tgt.rotations[idx];

        double best_pos = std::numeric_limits<double>::infinity();
        double best_rot = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < latest.size(); ++i) {
            double pe = (latest.component(i).mean().head(3) - truth_pos).norm();
            if (pe < best_pos) {
                best_pos = pe;
                best_rot = rotation_error_3d(
                    Eigen::Matrix3d(latest.component(i).rotation()), truth_R);
            }
        }

        pos_errors.push_back(best_pos);
        rot_errors.push_back(best_rot);

        bool good = (k >= warmup) && best_pos < pos_thresh && best_rot < rot_thresh;
        if (good) converged_steps++;

        std::cout << std::setw(5) << k
                  << std::setw(10) << measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(3) << best_pos
                  << std::setw(12) << best_rot
                  << std::setw(10) << ekf_ptr->last_icp_iterations()
                  << "  |";
        for (std::size_t i = 0; i < phd.intensity().size(); ++i) {
            std::cout << " " << std::setprecision(4) << phd.intensity().weight(i);
        }
        std::cout << "\n";
    }

    std::cout << "Converged steps (k>=" << warmup
              << ", pos<" << pos_thresh << ", rot<" << rot_thresh << "): "
              << converged_steps << " / " << (num_steps - warmup) << "\n";

    EXPECT_GE(converged_steps, 10)
        << "TM-PHD 3D should track the plane's position and rotation";

#ifdef BREW_ENABLE_PLOTTING
    std::filesystem::create_directories("output");

    // --- Figure 1: 3D tracking plot with wireframe + sampled points ---
    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        // Load STL triangles for wireframe rendering (centered same as template)
        Eigen::MatrixXd tri_verts = template_matching::load_stl_triangles("Plane.stl");
        tri_verts.colwise() -= centroid;
        const int num_tris = static_cast<int>(tri_verts.cols()) / 3;

        // Plot stride: show template shape every plot_stride steps
        const int plot_stride = 10;

        for (int k = 0; k < num_steps; k += plot_stride) {
            const auto& tgt = truth_targets[0];
            int idx = k - tgt.birth_time;
            Eigen::Vector3d pos = tgt.states[idx].head(3);
            const Eigen::Matrix3d& R = tgt.rotations[idx];

            // Batch wireframe: chain all triangles into one polyline
            // Each triangle contributes v0,v1,v2,v0 — connecting lines between
            // triangles are thin and barely visible at 0.3 width
            std::vector<double> wx, wy, wz;
            wx.reserve(4 * num_tris);
            wy.reserve(4 * num_tris);
            wz.reserve(4 * num_tris);
            for (int t = 0; t < num_tris; ++t) {
                Eigen::Vector3d v0 = R * tri_verts.col(3 * t + 0) + pos;
                Eigen::Vector3d v1 = R * tri_verts.col(3 * t + 1) + pos;
                Eigen::Vector3d v2 = R * tri_verts.col(3 * t + 2) + pos;
                wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                wx.push_back(v1(0)); wy.push_back(v1(1)); wz.push_back(v1(2));
                wx.push_back(v2(0)); wy.push_back(v2(1)); wz.push_back(v2(2));
                wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
            }
            auto wl = ax->plot3(wx, wy, wz);
            wl->color({0.f, 0.6f, 0.6f, 0.6f});
            wl->line_width(0.3f);

            // Draw sampled template points at this pose
            Eigen::MatrixXd pts = R * templ->points();
            pts.colwise() += pos;
            std::vector<double> sx(pts.cols()), sy(pts.cols()), sz(pts.cols());
            for (int j = 0; j < pts.cols(); ++j) {
                sx[j] = pts(0, j); sy[j] = pts(1, j); sz[j] = pts(2, j);
            }
            auto sp = ax->plot3(sx, sy, sz, ".");
            sp->color({0.f, 0.f, 0.f, 0.f});
            sp->marker_size(1.0f);
        }

        // Measurement points in red at each plotted timestep
        for (int k = 0; k < num_steps; k += plot_stride) {
            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(3.0f);
        }

        // Truth trajectory in black (3D line)
        {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[0].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

        // Estimated positions in color (3D scatter)
        auto est_color = brew::plot_utils::lines_color(0);
        std::vector<double> ex, ey, ez;
        for (int k = 0; k < num_steps; ++k) {
            const auto& latest = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < latest.size(); ++i) {
                ex.push_back(latest.component(i).mean()(0));
                ey.push_back(latest.component(i).mean()(1));
                ez.push_back(latest.component(i).mean()(2));
            }
        }
        if (!ex.empty()) {
            auto ep = ax->plot3(ex, ey, ez, "o");
            ep->color(est_color);
            ep->marker_size(5.0f);
            ep->marker_face_color(est_color);
            ep->marker_face(true);
        }

        ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
        ax->x_axis().label_font_size(18);
        ax->y_axis().label_font_size(18);
        ax->z_axis().label_font_size(18);
        ax->view(45.f, 30.f);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_plane.png");
    }

    // --- Figure 2: Convergence ---
    {
        auto fig = matplot::figure(true);
        fig->width(1200);
        fig->height(500);

        std::vector<double> steps(num_steps);
        for (int k = 0; k < num_steps; ++k) steps[k] = static_cast<double>(k);

        auto ax1 = matplot::subplot(fig, 1, 2, 0);
        ax1->hold(true);
        auto lp = ax1->plot(steps, pos_errors);
        lp->display_name("Position Error");
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto lr = ax2->plot(steps, rot_errors);
        lr->display_name("Rotation Error");
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_plane_convergence.png");
    }
#endif
}
