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
#include "brew/measurement_sampling/measurement_sampling.hpp"

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
    std::mt19937& rng)
{
    std::normal_distribution<double> noise(0.0, point_noise_std);

    std::vector<Eigen::MatrixXd> clouds;

    for (const auto& tgt : targets) {
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            {
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
    ekf->set_rotation_process_noise(0.2 * Eigen::MatrixXd::Identity(1, 1));
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
    ekf->set_rotation_process_noise(0.2 * Eigen::MatrixXd::Identity(1, 1));
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
    Eigen::MatrixXd R_empty;  // empty rotation — triggers PCA-ICP for cold start
    std::vector<int> pos_indices = {0, 1};

    // Both births centered at approximate scene center
    Eigen::VectorXd b0(4);
    b0 << 25.0, 15.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TemplatePose>(b0, birth_cov, R_empty, templ_rect, pos_indices), weight);
    birth->add_component(
        std::make_unique<models::TemplatePose>(b0, birth_cov, R_empty, templ_tri, pos_indices), weight);
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
    Eigen::MatrixXd R_empty;  // empty rotation — triggers PCA-ICP for cold start
    std::vector<int> pos_indices = {0, 1};

    Eigen::VectorXd b0(4);
    b0 << 25.0, 15.0, 0.0, 0.0;

    auto tp1 = models::TemplatePose(b0, birth_cov, R_empty, templ_rect, pos_indices);
    auto tp2 = models::TemplatePose(b0, birth_cov, R_empty, templ_tri, pos_indices);

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
    std::mt19937& rng)
{
    std::normal_distribution<double> noise(0.0, point_noise_std);

    std::vector<Eigen::MatrixXd> clouds;

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& tgt = targets[ti];
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            {
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
    double point_noise_std = 0.0001;  // low noise — this is an algorithm test
    double p_detect = 0.70;          // perfect detection for clean test

    std::shared_ptr<template_matching::PointCloud> templ_rect;  // rectangle
    std::shared_ptr<template_matching::PointCloud> templ_tri;     // L-shape
    std::vector<std::shared_ptr<template_matching::PointCloud>> target_templates;  // per-target
    std::shared_ptr<dynamics::SingleIntegrator> dyn;
    std::shared_ptr<template_matching::PointToPointIcp> icp;
    std::vector<TmTruthTarget> truth_targets;
    std::vector<Eigen::MatrixXd> measurements;

    void setup() {
        templ_rect = std::make_shared<template_matching::PointCloud>(
            measurement_sampling::sample_rectangle(3.0, 1.5, 20).points());
        templ_tri = std::make_shared<template_matching::PointCloud>(
            measurement_sampling::sample_triangle(3.0, 2.5, 20).points());
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
                truth_targets, target_templates, k, point_noise_std, rng);
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

// Print weights sorted highest to lowest, capped at max_show
template <typename T>
void print_sorted_weights(const models::Mixture<T>& mix, std::size_t max_show = 8) {
    std::vector<double> w(mix.size());
    for (std::size_t i = 0; i < mix.size(); ++i) w[i] = mix.weight(i);
    std::sort(w.begin(), w.end(), std::greater<double>());
    std::size_t n = std::min(w.size(), max_show);
    std::cout << std::defaultfloat << std::setprecision(4);
    for (std::size_t i = 0; i < n; ++i) std::cout << " " << w[i];
    if (w.size() > max_show) std::cout << " ...";
}

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
        print_sorted_weights(phd.intensity());
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
        auto la = ax1->plot(steps, pa); la->display_name("Target A"); la->line_width(2.0f);
        auto lb = ax1->plot(steps, pb); lb->display_name("Target B"); lb->line_width(2.0f);
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra_l = ax2->plot(steps, ra); ra_l->display_name("Target A"); ra_l->line_width(2.0f);
        auto rb_l = ax2->plot(steps, rb); rb_l->display_name("Target B"); rb_l->line_width(2.0f);
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
        print_sorted_weights(phd.intensity());
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
        auto la = ax1->plot(steps, pa); la->display_name("Target A"); la->line_width(2.0f);
        auto lb = ax1->plot(steps, pb); lb->display_name("Target B"); lb->line_width(2.0f);
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra_l = ax2->plot(steps, ra); ra_l->display_name("Target A"); ra_l->line_width(2.0f);
        auto rb_l = ax2->plot(steps, rb); rb_l->display_name("Target B"); rb_l->line_width(2.0f);
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

// Rotation matrix that aligns body x-axis with a velocity vector.
// Assumes body x = forward, z = up convention. Returns proper rotation (det=+1).
Eigen::Matrix3d facing_rotation(const Eigen::Vector3d& velocity) {
    Eigen::Vector3d fwd = velocity.normalized();
    // Choose an up hint that isn't parallel to fwd
    Eigen::Vector3d up_hint = Eigen::Vector3d::UnitZ();
    if (std::abs(fwd.dot(up_hint)) > 0.99) {
        up_hint = Eigen::Vector3d::UnitY();
    }
    Eigen::Vector3d right = up_hint.cross(fwd).normalized();
    Eigen::Vector3d up = fwd.cross(right).normalized();
    Eigen::Matrix3d R;
    R.col(0) = fwd;
    R.col(1) = right;
    R.col(2) = up;
    return R;
}

// Compute rotation that aligns a point cloud's principal axes with XYZ.
// Longest axis → X (forward), middle → Y (span), shortest → Z (up).
// Points must already be centered.
Eigen::Matrix3d pca_alignment(const Eigen::MatrixXd& pts) {
    Eigen::Matrix3d cov = (pts * pts.transpose()) / static_cast<double>(pts.cols());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d R;
    R.col(0) = solver.eigenvectors().col(2);  // largest  → forward (X)
    R.col(1) = solver.eigenvectors().col(1);  // middle   → span (Y)
    R.col(2) = solver.eigenvectors().col(0);  // smallest → up (Z)
    if (R.determinant() < 0) R.col(2) = -R.col(2);
    return R.transpose();  // maps PCA axes → standard axes
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

// Möller–Trumbore ray-triangle intersection.
// Returns true if ray hits triangle at parameter t in (eps, max_t - eps).
bool ray_triangle_intersect(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    const Eigen::Vector3d& v0,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    double max_t,
    double eps = 1e-6)
{
    Eigen::Vector3d e1 = v1 - v0;
    Eigen::Vector3d e2 = v2 - v0;
    Eigen::Vector3d h = direction.cross(e2);
    double a = e1.dot(h);
    if (std::abs(a) < eps) return false;

    double f = 1.0 / a;
    Eigen::Vector3d s = origin - v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;

    Eigen::Vector3d q = s.cross(e1);
    double v = f * direction.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;

    double t = f * e2.dot(q);
    return t > eps && t < max_t - eps;
}

// Check if a surface point is visible from a sensor via ray tracing.
// Backface culling + occlusion test against mesh triangles.
bool is_point_visible(
    const Eigen::Vector3d& world_pt,
    const Eigen::Vector3d& world_normal,
    const Eigen::Vector3d& sensor_pos,
    const Eigen::MatrixXd& world_tris,  // 3 x (3*num_tris)
    int num_tris)
{
    Eigen::Vector3d to_sensor = sensor_pos - world_pt;
    double dist = to_sensor.norm();
    if (dist < 1e-10) return false;

    // Backface: normal must face toward sensor
    if (world_normal.dot(to_sensor) < 0.0) return false;

    // Occlusion: cast ray from point toward sensor, offset origin along
    // the outward normal to avoid self-intersection with the source triangle
    // and its neighbors.
    Eigen::Vector3d origin = world_pt + 0.01 * world_normal;
    Eigen::Vector3d dir = (sensor_pos - origin).normalized();
    double ray_dist = (sensor_pos - origin).norm();
    for (int t = 0; t < num_tris; ++t) {
        if (ray_triangle_intersect(origin, dir,
                world_tris.col(3*t), world_tris.col(3*t+1), world_tris.col(3*t+2),
                ray_dist)) {
            return false;
        }
    }
    return true;
}

// Generate 3D point cloud measurements by sampling mesh faces.
// Samples points from triangle surfaces (area-weighted barycentric),
// then filters by ray-traced visibility from the sensor position.
// mesh_triangles: per-target body-frame triangle vertices (3 x 3*N_tri).
// num_candidates: how many face points to sample before visibility filtering.
// num_meas_pts: subsample visible points to this many (0 = use all visible).
Eigen::MatrixXd generate_tm_measurements_3d(
    const std::vector<TmTruthTarget3D>& targets,
    const std::vector<Eigen::MatrixXd>& mesh_triangles,
    const Eigen::Vector3d& sensor_pos,
    int timestep, double point_noise_std,
    std::mt19937& rng,
    int num_candidates = 1000,
    int num_meas_pts = 0)
{
    std::normal_distribution<double> noise(0.0, point_noise_std);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Eigen::MatrixXd> clouds;

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& tgt = targets[ti];
        if (timestep >= tgt.birth_time && timestep <= tgt.death_time) {
            int idx = timestep - tgt.birth_time;
            Eigen::Vector3d pos = tgt.states[idx].head(3);
            const Eigen::Matrix3d& R = tgt.rotations[idx];

            const Eigen::MatrixXd& body_tris = mesh_triangles[ti];
            int num_tris = static_cast<int>(body_tris.cols()) / 3;

            // Build cumulative area distribution for face sampling
            std::vector<double> cum_area(num_tris);
            for (int t = 0; t < num_tris; ++t) {
                Eigen::Vector3d e1 = body_tris.col(3*t+1) - body_tris.col(3*t);
                Eigen::Vector3d e2 = body_tris.col(3*t+2) - body_tris.col(3*t);
                double area = 0.5 * e1.cross(e2).norm();
                cum_area[t] = (t > 0 ? cum_area[t-1] : 0.0) + area;
            }
            double total_area = cum_area.back();

            // Transform mesh to world frame for occlusion testing
            Eigen::MatrixXd world_tris = R * body_tris;
            world_tris.colwise() += pos;

            // Sample candidate points from mesh faces and check visibility
            std::vector<Eigen::Vector3d> visible_pts;
            visible_pts.reserve(num_candidates);

            for (int c = 0; c < num_candidates; ++c) {
                // Pick a triangle proportional to area
                double r = uniform(rng) * total_area;
                auto it = std::lower_bound(cum_area.begin(), cum_area.end(), r);
                int tri_idx = static_cast<int>(std::distance(cum_area.begin(), it));
                tri_idx = std::min(tri_idx, num_tris - 1);

                // Sample point on triangle via barycentric coordinates
                double u = uniform(rng), v = uniform(rng);
                double su = std::sqrt(u);
                double b0 = 1.0 - su, b1 = v * su, b2 = 1.0 - b0 - b1;
                Eigen::Vector3d body_pt = b0 * body_tris.col(3*tri_idx)
                                        + b1 * body_tris.col(3*tri_idx+1)
                                        + b2 * body_tris.col(3*tri_idx+2);

                // Face normal
                Eigen::Vector3d e1 = body_tris.col(3*tri_idx+1) - body_tris.col(3*tri_idx);
                Eigen::Vector3d e2 = body_tris.col(3*tri_idx+2) - body_tris.col(3*tri_idx);
                Eigen::Vector3d n = e1.cross(e2);
                double len = n.norm();
                if (len < 1e-15) continue;
                n /= len;

                // Transform to world frame
                Eigen::Vector3d world_pt = R * body_pt + pos;
                Eigen::Vector3d world_normal = R * n;

                if (is_point_visible(world_pt, world_normal, sensor_pos,
                                     world_tris, num_tris)) {
                    visible_pts.push_back(world_pt);
                }
            }

            if (visible_pts.empty()) continue;

            // Subsample visible points if requested
            if (num_meas_pts > 0 && num_meas_pts < static_cast<int>(visible_pts.size())) {
                std::shuffle(visible_pts.begin(), visible_pts.end(), rng);
                visible_pts.resize(num_meas_pts);
            }

            Eigen::MatrixXd pts(3, static_cast<int>(visible_pts.size()));
            for (int j = 0; j < static_cast<int>(visible_pts.size()); ++j) {
                pts.col(j) = visible_pts[j];
                pts(0, j) += noise(rng);
                pts(1, j) += noise(rng);
                pts(2, j) += noise(rng);
            }
            clouds.push_back(pts);
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


TEST(TmPhd, Tracking3D_SingleTarget) {
    // --- Load template with PCA alignment (longest axis → X) ---
    auto load_template = [](const std::string& stl_file, int n_pts) {
        auto cloud = measurement_sampling::sample_stl(stl_file, n_pts);
        Eigen::Vector3d centroid = cloud.points().rowwise().mean();
        cloud.points().colwise() -= centroid;
        Eigen::Matrix3d R_align = pca_alignment(cloud.points());
        Eigen::MatrixXd aligned = R_align * cloud.points();
        // Ensure nose faces +X: flip 180° around Z if needed
        if (std::abs(aligned.row(0).minCoeff()) > aligned.row(0).maxCoeff()) {
            Eigen::Matrix3d flip_z = Eigen::Matrix3d::Identity();
            flip_z(0, 0) = -1.0;
            flip_z(1, 1) = -1.0;
            aligned = flip_z * aligned;
            R_align = flip_z * R_align;
        }
        // Ensure top faces +Z: flip 180° around X if needed
        if (std::abs(aligned.row(2).minCoeff()) > aligned.row(2).maxCoeff()) {
            Eigen::Matrix3d flip_x = Eigen::Matrix3d::Identity();
            flip_x(1, 1) = -1.0;
            flip_x(2, 2) = -1.0;
            aligned = flip_x * aligned;
            R_align = flip_x * R_align;
        }
        auto templ = std::make_shared<template_matching::PointCloud>(aligned);
        return std::make_tuple(templ, centroid, R_align);
    };

    auto [templ_jet, centroid_jet, R_align_jet] = load_template("Plane.stl", 1500);

    // --- Dynamics ---
    auto dyn = std::make_shared<dynamics::SingleIntegrator>(3);

    // --- ICP ---
    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 25;
    icp_params.tolerance = 1e-10;
    icp_params.sigma_sq = 0.001;

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    auto ekf = std::make_unique<filters::TmEkf>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(0.50 * Eigen::MatrixXd::Identity(3, 3));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.1 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);

    // --- Single target scenario ---
    const int num_steps = 10;
    const double dt = 0.2;
    const double point_noise_std = 0.0001;

    Eigen::VectorXd x0(6);
    x0 << 0.0, 0.0, 50.0, 10.0, 5.0, 0.3;
    Eigen::Matrix3d R0 = facing_rotation(x0.tail(3));
    Eigen::Vector3d omega(0.0, 0.0, 0.0);

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0, R0, omega, 0, num_steps - 1, dt, *dyn));

    // --- Mesh triangles for face-sampled measurements (same alignment as template) ---
    Eigen::MatrixXd tri_jet_body = measurement_sampling::load_stl_triangles("Plane.stl");
    tri_jet_body.colwise() -= centroid_jet;
    tri_jet_body = R_align_jet * tri_jet_body;

    std::vector<Eigen::MatrixXd> meas_triangles = {tri_jet_body};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {
        // Sensor directly below the target at ground level
        Eigen::Vector3d sensor_pos(truth_targets[0].states[k](0),
                                   truth_targets[0].states[k](1), 0.0);
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, meas_triangles, sensor_pos,
            k, point_noise_std, rng, 1000, 100);
    }

    // --- Birth model ---
    auto birth = std::make_unique<models::Mixture<models::TemplatePose>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 1500.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 500.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = 1.0 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_birth = Eigen::Matrix3d::Identity();
    std::vector<int> pos_indices = {0, 1, 2};

    Eigen::VectorXd b_mean(6);
    b_mean << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TemplatePose>(b_mean, birth_cov, R_birth, templ_jet, pos_indices), 0.01);

    // --- PHD filter --- 
    multi_target::PHD<models::TemplatePose> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::TemplatePose>>());
    phd.set_prob_detection(1.0);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-6);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(20);
    phd.set_extract_threshold(0.4); 
    phd.set_gate_threshold(100.0);
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(8.0, 3));

    // --- Run filter ---
    std::cout << "\n=== TM-PHD 3D Single Target (Jet) ===\n";
    std::cout << std::setw(5) << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est"
              << std::setw(12) << "pos_err"
              << std::setw(12) << "rot_err"
              << "  | weights\n";

    int converged_steps = 0;
    const int warmup = 5;
    const double pos_thresh = 5.0;
    const double rot_thresh = 0.5;

    std::vector<double> pos_errs, rot_errs;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);
        phd.correct(measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();
        const auto& tgt = truth_targets[0];
        Eigen::Vector3d truth_pos = tgt.states[k].head(3);
        const Eigen::Matrix3d& truth_R = tgt.rotations[k];

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

        pos_errs.push_back(best_pos);
        rot_errs.push_back(best_rot);

        bool good = (k >= warmup) && best_pos < pos_thresh && best_rot < rot_thresh;
        if (good) converged_steps++;

        std::cout << std::setw(5) << k
                  << std::setw(10) << measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(3) << best_pos
                  << std::setw(12) << best_rot
                  << "  |";
        print_sorted_weights(phd.intensity());
        std::cout << "\n";
    }

    std::cout << "Converged steps (k>=" << warmup
              << ", pos<" << pos_thresh << ", rot<" << rot_thresh << "): "
              << converged_steps << " / " << (num_steps - warmup) << "\n";

    EXPECT_GE(converged_steps, 1)
        << "TM-PHD 3D single target should track position and rotation";

#ifdef BREW_ENABLE_PLOTTING
    std::filesystem::create_directories("output");

    // Load STL triangles for wireframe rendering (same PCA alignment)
    Eigen::MatrixXd tri_verts = measurement_sampling::load_stl_triangles("Plane.stl");
    tri_verts.colwise() -= centroid_jet;
    tri_verts = R_align_jet * tri_verts;
    const int num_tris = static_cast<int>(tri_verts.cols()) / 3;

    const int plot_stride = 3;

    // --- Figure 1: 3D close-up — estimated wireframe vs measurements ---
    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        for (int k = 0; k < num_steps; k += plot_stride) {
            // Measurement points in red
            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(5.0f);

            // Estimated wireframe at estimated pose
            const auto& latest = *phd.extracted_mixtures()[k];
            if (latest.empty()) continue;
            const auto& est = latest.component(0);
            Eigen::Vector3d est_pos = est.mean().head(3);
            Eigen::Matrix3d est_R = Eigen::Matrix3d(est.rotation());

            std::vector<double> wx, wy, wz;
            for (int t = 0; t < num_tris; ++t) {
                Eigen::Vector3d v0 = est_R * tri_verts.col(3 * t + 0) + est_pos;
                Eigen::Vector3d v1 = est_R * tri_verts.col(3 * t + 1) + est_pos;
                Eigen::Vector3d v2 = est_R * tri_verts.col(3 * t + 2) + est_pos;
                wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                wx.push_back(v1(0)); wy.push_back(v1(1)); wz.push_back(v1(2));
                wx.push_back(v2(0)); wy.push_back(v2(1)); wz.push_back(v2(2));
                wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                // NaN break between triangles
                wx.push_back(std::nan("")); wy.push_back(std::nan("")); wz.push_back(std::nan(""));
            }
            auto wl = ax->plot3(wx, wy, wz);
            wl->color({0.f, 0.5f, 0.5f, 0.5f});
            wl->line_width(0.2f);
        }

        // Truth trajectory in black
        {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[0].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

        // Estimated means as markers along the trajectory
        {
            std::vector<double> ex, ey, ez;
            for (int k = 0; k < num_steps; ++k) {
                const auto& latest = *phd.extracted_mixtures()[k];
                if (latest.empty()) continue;
                ex.push_back(latest.component(0).mean()(0));
                ey.push_back(latest.component(0).mean()(1));
                ez.push_back(latest.component(0).mean()(2));
            }
            auto ep = ax->plot3(ex, ey, ez, "o");
            auto c = brew::plot_utils::lines_color(0);
            ep->color(c);
            ep->marker_size(5.0f);
            ep->marker_face_color(c);
            ep->marker_face(true);
        }

        ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
        ax->x_axis().label_font_size(18);
        ax->y_axis().label_font_size(18);
        ax->z_axis().label_font_size(18);
        ax->axes_aspect_ratio_auto(false);
        ax->axes_aspect_ratio(1.0f);
        ax->view(45.f, 30.f);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_single.png");
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
        auto la = ax1->plot(steps, pos_errs); la->display_name("Jet"); la->line_width(2.0f);
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra = ax2->plot(steps, rot_errs); ra->display_name("Jet"); ra->line_width(2.0f);
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_single_convergence.png");
    }
#endif
}


TEST(TmPhd, Tracking3D_PlaneAndJet) {
    // --- Load and prepare templates with PCA alignment ---
    auto load_template = [](const std::string& stl_file, int n_pts) {
        auto cloud = measurement_sampling::sample_stl(stl_file, n_pts);
        Eigen::Vector3d centroid = cloud.points().rowwise().mean();
        cloud.points().colwise() -= centroid;
        Eigen::Matrix3d R_align = pca_alignment(cloud.points());
        Eigen::MatrixXd aligned = R_align * cloud.points();
        // Ensure nose faces +X: flip 180° around Z if needed
        if (std::abs(aligned.row(0).minCoeff()) > aligned.row(0).maxCoeff()) {
            Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
            flip(0, 0) = -1.0;
            flip(1, 1) = -1.0;
            aligned = flip * aligned;
            R_align = flip * R_align;
        }
        auto templ = std::make_shared<template_matching::PointCloud>(aligned);
        return std::make_tuple(templ, centroid, R_align);
    };

    auto [templ_plane, centroid_plane, R_align_plane] = load_template("Plane.stl", 1500);
    auto [templ_jet,   centroid_jet,   R_align_jet]   = load_template("Jet.stl",   1500);

    std::cout << "\n--- Templates (PCA-aligned) ---\n";
    std::cout << "Plane: " << templ_plane->num_points() << " pts, bbox X["
              << templ_plane->points().row(0).minCoeff() << ", " << templ_plane->points().row(0).maxCoeff()
              << "] Y[" << templ_plane->points().row(1).minCoeff() << ", " << templ_plane->points().row(1).maxCoeff()
              << "] Z[" << templ_plane->points().row(2).minCoeff() << ", " << templ_plane->points().row(2).maxCoeff() << "]\n";
    std::cout << "Jet:   " << templ_jet->num_points() << " pts, bbox X["
              << templ_jet->points().row(0).minCoeff() << ", " << templ_jet->points().row(0).maxCoeff()
              << "] Y[" << templ_jet->points().row(1).minCoeff() << ", " << templ_jet->points().row(1).maxCoeff()
              << "] Z[" << templ_jet->points().row(2).minCoeff() << ", " << templ_jet->points().row(2).maxCoeff() << "]\n";

    // --- Dynamics ---
    auto dyn = std::make_shared<dynamics::SingleIntegrator>(3);

    // --- ICP (shared params, per-template PCA) ---
    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 25;
    icp_params.tolerance = 1e-6;
    icp_params.sigma_sq = 0.1;

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    // PCA wrapping is automatic per template — just set the inner ICP
    auto ekf = std::make_unique<filters::TmEkf>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(3, 3));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.05 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);

    // --- Scenario: two targets, facing their velocity, moderate separation ---
    const int num_steps = 25;
    const double dt = 0.2;
    const double point_noise_std = 0.0001;
    const double p_detect = 0.95;

    // Target A: Plane — heading northeast
    Eigen::VectorXd x0_plane(6);
    x0_plane << 0.0, 0.0, 50.0, 15.0, 10.0, 0.5;
    Eigen::Matrix3d R0_plane = facing_rotation(x0_plane.tail(3));
    Eigen::Vector3d omega_plane(0.0, 0.0, 0.0);

    // Target B: Jet — heading east, nearby but different altitude
    Eigen::VectorXd x0_jet(6);
    x0_jet << 50.0, 80.0, 80.0, 20.0, -5.0, -0.3;
    Eigen::Matrix3d R0_jet = facing_rotation(x0_jet.tail(3));
    Eigen::Vector3d omega_jet(0.0, 0.0, 0.0);

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0_plane, R0_plane, omega_plane, 0, num_steps - 1, dt, *dyn));
    truth_targets.push_back(make_tm_target_3d(x0_jet,   R0_jet,   omega_jet,   0, num_steps - 1, dt, *dyn));

    // --- Mesh triangles for face-sampled measurements ---
    Eigen::MatrixXd tri_plane_body = measurement_sampling::load_stl_triangles("Plane.stl");
    tri_plane_body.colwise() -= centroid_plane;
    tri_plane_body = R_align_plane * tri_plane_body;
    Eigen::MatrixXd tri_jet_body = measurement_sampling::load_stl_triangles("Jet.stl");
    tri_jet_body.colwise() -= centroid_jet;
    tri_jet_body = R_align_jet * tri_jet_body;

    std::vector<Eigen::MatrixXd> meas_triangles = {tri_plane_body, tri_jet_body};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {
        // Sensor at ground level below the midpoint of all targets
        Eigen::Vector3d mid = Eigen::Vector3d::Zero();
        int n_active = 0;
        for (const auto& tgt : truth_targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                mid += tgt.states[k - tgt.birth_time].head(3);
                ++n_active;
            }
        }
        if (n_active > 0) mid /= static_cast<double>(n_active);
        Eigen::Vector3d sensor_pos(mid(0), mid(1), 0.0);
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, meas_triangles, sensor_pos,
            k, point_noise_std, rng, 1000, 100);
    }

    // --- Birth model: one component per template ---
    auto birth = std::make_unique<models::Mixture<models::TemplatePose>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 1500.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 500.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = 1.0 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_birth = Eigen::Matrix3d::Identity();
    std::vector<int> pos_indices = {0, 1, 2};

    // Single general birth mean
    Eigen::VectorXd b_mean(6);
    b_mean << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    double w_b = 0.001;
    birth->add_component(
        std::make_unique<models::TemplatePose>(b_mean, birth_cov, R_birth, templ_plane, pos_indices), w_b);
    birth->add_component(
        std::make_unique<models::TemplatePose>(b_mean, birth_cov, R_birth, templ_jet, pos_indices), w_b);

    // --- PHD filter ---
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
    phd.set_max_components(50);
    phd.set_extract_threshold(0.4); 
    phd.set_gate_threshold(100.0);
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(8.0, 3));

    // --- Run filter ---
    std::cout << "\n=== TM-PHD 3D Multi-Template Tracking (Plane + Jet) ===\n";
    std::cout << std::setw(5) << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est"
              << std::setw(12) << "pos_A"
              << std::setw(12) << "rot_A"
              << std::setw(12) << "pos_B"
              << std::setw(12) << "rot_B"
              << "  | weights\n";

    int converged_steps = 0;
    const int warmup = 5;
    const double pos_thresh = 5.0;
    const double rot_thresh = 0.5;

    std::vector<double> pos_A, rot_A, pos_B, rot_B;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);
        phd.correct(measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();

        // Compute best error for each truth target
        double best_pos[2] = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
        double best_rot[2] = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};

        for (int t = 0; t < 2; ++t) {
            const auto& tgt = truth_targets[t];
            int idx = k - tgt.birth_time;
            Eigen::Vector3d truth_pos = tgt.states[idx].head(3);
            const Eigen::Matrix3d& truth_R = tgt.rotations[idx];

            for (std::size_t i = 0; i < latest.size(); ++i) {
                double pe = (latest.component(i).mean().head(3) - truth_pos).norm();
                if (pe < best_pos[t]) {
                    best_pos[t] = pe;
                    best_rot[t] = rotation_error_3d(
                        Eigen::Matrix3d(latest.component(i).rotation()), truth_R);
                }
            }
        }

        pos_A.push_back(best_pos[0]); rot_A.push_back(best_rot[0]);
        pos_B.push_back(best_pos[1]); rot_B.push_back(best_rot[1]);

        bool good = (k >= warmup)
                   && best_pos[0] < pos_thresh && best_rot[0] < rot_thresh
                   && best_pos[1] < pos_thresh && best_rot[1] < rot_thresh;
        if (good) converged_steps++;

        std::cout << std::setw(5) << k
                  << std::setw(10) << measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(3) << best_pos[0]
                  << std::setw(12) << best_rot[0]
                  << std::setw(12) << best_pos[1]
                  << std::setw(12) << best_rot[1]
                  << "  |";
        print_sorted_weights(phd.intensity());
        std::cout << "\n";

    }

    std::cout << "Converged steps (k>=" << warmup
              << ", pos<" << pos_thresh << ", rot<" << rot_thresh << "): "
              << converged_steps << " / " << (num_steps - warmup) << "\n";

    EXPECT_GE(converged_steps, 10)
        << "TM-PHD 3D should track both targets' position and rotation";

#ifdef BREW_ENABLE_PLOTTING
    std::filesystem::create_directories("output");

    // --- Figure 1: 3D tracking plot ---
    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        const int plot_stride = 3;

        // Load STL triangles (same PCA alignment as templates)
        Eigen::MatrixXd tri_plane = measurement_sampling::load_stl_triangles("Plane.stl");
        tri_plane.colwise() -= centroid_plane;
        tri_plane = R_align_plane * tri_plane;
        const int n_tri_plane = static_cast<int>(tri_plane.cols()) / 3;

        Eigen::MatrixXd tri_jet = measurement_sampling::load_stl_triangles("Jet.stl");
        tri_jet.colwise() -= centroid_jet;
        tri_jet = R_align_jet * tri_jet;
        const int n_tri_jet = static_cast<int>(tri_jet.cols()) / 3;

        auto get_tri_data = [&](const models::TemplatePose& est)
            -> std::pair<const Eigen::MatrixXd*, int> {
            if (&est.get_template() == templ_plane.get())
                return {&tri_plane, n_tri_plane};
            return {&tri_jet, n_tri_jet};
        };

        for (int k = 0; k < num_steps; k += plot_stride) {
            // Measurement points in red
            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(3.0f);

            // Estimated wireframes at estimated pose
            const auto& latest = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < latest.size(); ++i) {
                const auto& est = latest.component(i);
                Eigen::Vector3d est_pos = est.mean().head(3);
                Eigen::Matrix3d est_R = Eigen::Matrix3d(est.rotation());
                auto [tri_verts, num_tris] = get_tri_data(est);

                std::vector<double> wx, wy, wz;
                for (int t = 0; t < num_tris; ++t) {
                    Eigen::Vector3d v0 = est_R * tri_verts->col(3 * t + 0) + est_pos;
                    Eigen::Vector3d v1 = est_R * tri_verts->col(3 * t + 1) + est_pos;
                    Eigen::Vector3d v2 = est_R * tri_verts->col(3 * t + 2) + est_pos;
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(v1(0)); wy.push_back(v1(1)); wz.push_back(v1(2));
                    wx.push_back(v2(0)); wy.push_back(v2(1)); wz.push_back(v2(2));
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(std::nan("")); wy.push_back(std::nan("")); wz.push_back(std::nan(""));
                }
                auto wl = ax->plot3(wx, wy, wz);
                wl->color({0.f, 0.7f, 0.7f, 0.7f});
                wl->line_width(0.2f);
            }
        }

        // Truth trajectories in black
        for (int t = 0; t < 2; ++t) {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[t].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

        // Estimated means as markers along the trajectory
        for (int k = 0; k < num_steps; ++k) {
            const auto& latest = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < latest.size(); ++i) {
                auto c = brew::plot_utils::lines_color(static_cast<int>(i));
                std::vector<double> ex = {latest.component(i).mean()(0)};
                std::vector<double> ey = {latest.component(i).mean()(1)};
                std::vector<double> ez = {latest.component(i).mean()(2)};
                auto ep = ax->plot3(ex, ey, ez, "o");
                ep->color(c);
                ep->marker_size(4.0f);
                ep->marker_face_color(c);
                ep->marker_face(true);
            }
        }

        ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
        ax->x_axis().label_font_size(18);
        ax->y_axis().label_font_size(18);
        ax->z_axis().label_font_size(18);
        ax->view(45.f, 30.f);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_multi.png");
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
        auto la = ax1->plot(steps, pos_A); la->display_name("Plane"); la->line_width(2.0f);
        auto lb = ax1->plot(steps, pos_B); lb->display_name("Jet"); lb->line_width(2.0f);
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra = ax2->plot(steps, rot_A); ra->display_name("Plane"); ra->line_width(2.0f);
        auto rb = ax2->plot(steps, rot_B); rb->display_name("Jet"); rb->line_width(2.0f);
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_multi_convergence.png");
    }
#endif
}


// ============================================================
// Trajectory TM-PHD 3D: trajectory + shape tracking test
// ============================================================

TEST(TmPhd, TrajectoryTracking3D_PlaneAndJet) {
    // --- Load and prepare templates with PCA alignment ---
    auto load_template = [](const std::string& stl_file, int n_pts) {
        auto cloud = measurement_sampling::sample_stl(stl_file, n_pts);
        Eigen::Vector3d centroid = cloud.points().rowwise().mean();
        cloud.points().colwise() -= centroid;
        Eigen::Matrix3d R_align = pca_alignment(cloud.points());
        Eigen::MatrixXd aligned = R_align * cloud.points();
        // Ensure nose faces +X: flip 180° around Z if needed
        if (std::abs(aligned.row(0).minCoeff()) > aligned.row(0).maxCoeff()) {
            Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
            flip(0, 0) = -1.0;
            flip(1, 1) = -1.0;
            aligned = flip * aligned;
            R_align = flip * R_align;
        }
        auto templ = std::make_shared<template_matching::PointCloud>(aligned);
        return std::make_tuple(templ, centroid, R_align);
    };

    auto [templ_plane, centroid_plane, R_align_plane] = load_template("Plane.stl", 1500);
    auto [templ_jet,   centroid_jet,   R_align_jet]   = load_template("Jet.stl",   1500);

    // --- Dynamics ---
    auto dyn = std::make_shared<dynamics::SingleIntegrator>(3);

    // --- ICP ---
    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 25;
    icp_params.tolerance = 1e-6;
    icp_params.sigma_sq = 0.1;

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    // --- Trajectory TM-EKF ---
    auto ekf = std::make_unique<filters::TrajectoryTmEkf>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(3, 3));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.05 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);
    ekf->set_window_size(5);

    // --- Scenario (same as non-trajectory 3D test) ---
    const int num_steps = 25;
    const double dt = 0.2;
    const double point_noise_std = 0.0001;
    const double p_detect = 0.95;

    Eigen::VectorXd x0_plane(6);
    x0_plane << 0.0, 0.0, 50.0, 15.0, 10.0, 0.5;
    Eigen::Matrix3d R0_plane = facing_rotation(x0_plane.tail(3));
    Eigen::Vector3d omega_plane(0.0, 0.0, 0.0);

    Eigen::VectorXd x0_jet(6);
    x0_jet << 50.0, 80.0, 80.0, 20.0, -5.0, -0.3;
    Eigen::Matrix3d R0_jet = facing_rotation(x0_jet.tail(3));
    Eigen::Vector3d omega_jet(0.0, 0.0, 0.0);

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0_plane, R0_plane, omega_plane, 0, num_steps - 1, dt, *dyn));
    truth_targets.push_back(make_tm_target_3d(x0_jet,   R0_jet,   omega_jet,   0, num_steps - 1, dt, *dyn));

    // --- Mesh triangles for face-sampled measurements ---
    Eigen::MatrixXd tri_plane_body = measurement_sampling::load_stl_triangles("Plane.stl");
    tri_plane_body.colwise() -= centroid_plane;
    tri_plane_body = R_align_plane * tri_plane_body;
    Eigen::MatrixXd tri_jet_body = measurement_sampling::load_stl_triangles("Jet.stl");
    tri_jet_body.colwise() -= centroid_jet;
    tri_jet_body = R_align_jet * tri_jet_body;

    std::vector<Eigen::MatrixXd> meas_triangles = {tri_plane_body, tri_jet_body};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {
        // Sensor at ground level below the midpoint of all targets
        Eigen::Vector3d mid = Eigen::Vector3d::Zero();
        int n_active = 0;
        for (const auto& tgt : truth_targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                mid += tgt.states[k - tgt.birth_time].head(3);
                ++n_active;
            }
        }
        if (n_active > 0) mid /= static_cast<double>(n_active);
        Eigen::Vector3d sensor_pos(mid(0), mid(1), 0.0);
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, meas_triangles, sensor_pos,
            k, point_noise_std, rng, 1000, 100);
    }

    // --- Birth model ---
    auto birth = std::make_unique<models::Mixture<models::Trajectory<models::TemplatePose>>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 1500.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 500.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = 1.0 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_birth = Eigen::Matrix3d::Identity();
    std::vector<int> pos_indices = {0, 1, 2};
    const int sd = 6;  // translational state dim

    Eigen::VectorXd b_mean(sd);
    b_mean << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    auto make_traj_birth = [&](std::shared_ptr<template_matching::PointCloud> templ) {
        auto tp = models::TemplatePose(b_mean, birth_cov, R_birth, templ, pos_indices);
        auto traj = std::make_unique<models::Trajectory<models::TemplatePose>>();
        traj->state_dim = sd;
        traj->mean() = tp.mean();
        traj->covariance() = tp.covariance().topLeftCorner(sd, sd);
        traj->history().push_back(std::move(tp));
        return traj;
    };

    double w_b = 0.001;

    birth->add_component(make_traj_birth(templ_plane), w_b);
    birth->add_component(make_traj_birth(templ_jet), w_b);

    // --- PHD filter ---
    multi_target::PHD<models::Trajectory<models::TemplatePose>> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(
        std::make_unique<models::Mixture<models::Trajectory<models::TemplatePose>>>());
    phd.set_prob_detection(p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1.0);
    phd.set_clutter_density(1e-6);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(50);
    phd.set_extract_threshold(0.4); 
    phd.set_gate_threshold(100.0);
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(8.0, 3));

    // --- Run filter ---
    std::cout << "\n=== Trajectory TM-PHD 3D Multi-Template Tracking (Plane + Jet) ===\n";
    std::cout << std::setw(5) << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est"
              << std::setw(12) << "pos_A"
              << std::setw(12) << "rot_A"
              << std::setw(12) << "pos_B"
              << std::setw(12) << "rot_B"
              << "  | weights\n";

    int converged_steps = 0;
    const int warmup = 5;
    const double pos_thresh = 5.0;
    const double rot_thresh = 0.5;

    std::vector<double> pos_A, rot_A, pos_B, rot_B;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);
        phd.correct(measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();

        double best_pos[2] = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
        double best_rot[2] = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};

        for (int t = 0; t < 2; ++t) {
            const auto& tgt = truth_targets[t];
            int idx = k - tgt.birth_time;
            Eigen::Vector3d truth_pos = tgt.states[idx].head(3);
            const Eigen::Matrix3d& truth_R = tgt.rotations[idx];

            for (std::size_t i = 0; i < latest.size(); ++i) {
                const auto& tp = latest.component(i).current();
                double pe = (tp.mean().head(3) - truth_pos).norm();
                if (pe < best_pos[t]) {
                    best_pos[t] = pe;
                    best_rot[t] = rotation_error_3d(
                        Eigen::Matrix3d(tp.rotation()), truth_R);
                }
            }
        }

        pos_A.push_back(best_pos[0]); rot_A.push_back(best_rot[0]);
        pos_B.push_back(best_pos[1]); rot_B.push_back(best_rot[1]);

        bool good = (k >= warmup)
                   && best_pos[0] < pos_thresh && best_rot[0] < rot_thresh
                   && best_pos[1] < pos_thresh && best_rot[1] < rot_thresh;
        if (good) converged_steps++;

        std::cout << std::setw(5) << k
                  << std::setw(10) << measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size()
                  << std::setw(12) << std::fixed << std::setprecision(3) << best_pos[0]
                  << std::setw(12) << best_rot[0]
                  << std::setw(12) << best_pos[1]
                  << std::setw(12) << best_rot[1]
                  << "  |";
        print_sorted_weights(phd.intensity());
        std::cout << "\n";
    }

    std::cout << "Converged steps (k>=" << warmup
              << ", pos<" << pos_thresh << ", rot<" << rot_thresh << "): "
              << converged_steps << " / " << (num_steps - warmup) << "\n";

    EXPECT_GE(converged_steps, 10)
        << "Trajectory TM-PHD 3D should track both targets' position and rotation";

#ifdef BREW_ENABLE_PLOTTING
    std::filesystem::create_directories("output");

    // --- Figure 1: 3D tracking plot with trajectory history ---
    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        const int plot_stride = 3;

        // Load STL triangles (same PCA alignment as templates)
        Eigen::MatrixXd tri_plane = measurement_sampling::load_stl_triangles("Plane.stl");
        tri_plane.colwise() -= centroid_plane;
        tri_plane = R_align_plane * tri_plane;
        const int n_tri_plane = static_cast<int>(tri_plane.cols()) / 3;

        Eigen::MatrixXd tri_jet = measurement_sampling::load_stl_triangles("Jet.stl");
        tri_jet.colwise() -= centroid_jet;
        tri_jet = R_align_jet * tri_jet;
        const int n_tri_jet = static_cast<int>(tri_jet.cols()) / 3;

        auto get_tri_data = [&](const models::TemplatePose& est)
            -> std::pair<const Eigen::MatrixXd*, int> {
            if (&est.get_template() == templ_plane.get())
                return {&tri_plane, n_tri_plane};
            return {&tri_jet, n_tri_jet};
        };

        for (int k = 0; k < num_steps; k += plot_stride) {
            // Measurement points in red
            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(3.0f);

            // Estimated wireframes at estimated pose
            const auto& latest = *phd.extracted_mixtures()[k];
            for (std::size_t i = 0; i < latest.size(); ++i) {
                const auto& tp = latest.component(i).current();
                Eigen::Vector3d est_pos = tp.mean().head(3);
                Eigen::Matrix3d est_R = Eigen::Matrix3d(tp.rotation());
                auto [tri_verts, num_tris] = get_tri_data(tp);

                std::vector<double> wx, wy, wz;
                for (int ti = 0; ti < num_tris; ++ti) {
                    Eigen::Vector3d v0 = est_R * tri_verts->col(3 * ti + 0) + est_pos;
                    Eigen::Vector3d v1 = est_R * tri_verts->col(3 * ti + 1) + est_pos;
                    Eigen::Vector3d v2 = est_R * tri_verts->col(3 * ti + 2) + est_pos;
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(v1(0)); wy.push_back(v1(1)); wz.push_back(v1(2));
                    wx.push_back(v2(0)); wy.push_back(v2(1)); wz.push_back(v2(2));
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(std::nan("")); wy.push_back(std::nan("")); wz.push_back(std::nan(""));
                }
                auto wl = ax->plot3(wx, wy, wz);
                wl->color({0.f, 0.7f, 0.7f, 0.7f});
                wl->line_width(0.2f);
            }
        }

        // Truth trajectories in black
        for (int t = 0; t < 2; ++t) {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[t].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

        // Estimated means as markers from trajectory history
        const auto& final_mix = *phd.extracted_mixtures().back();
        for (std::size_t i = 0; i < final_mix.size(); ++i) {
            const auto& traj = final_mix.component(i);
            const auto& hist = traj.history();
            if (hist.empty()) continue;
            std::vector<double> ex, ey, ez;
            for (const auto& tp : hist) {
                ex.push_back(tp.mean()(0));
                ey.push_back(tp.mean()(1));
                ez.push_back(tp.mean()(2));
            }
            auto c = brew::plot_utils::lines_color(static_cast<int>(i));
            auto el = ax->plot3(ex, ey, ez, "o-");
            el->color(c);
            el->line_width(1.5f);
            el->marker_size(4.0f);
            el->marker_face_color(c);
            el->marker_face(true);
        }

        ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
        ax->x_axis().label_font_size(18);
        ax->y_axis().label_font_size(18);
        ax->z_axis().label_font_size(18);
        ax->view(45.f, 30.f);

        brew::plot_utils::save_figure(fig, "output/trajectory_tm_phd_3d_multi.png");
    }

    // --- Figure 2: Convergence from smoothed trajectory history ---
    {
        auto fig = matplot::figure(true);
        fig->width(1200);
        fig->height(500);

        // Compute errors from the final extraction's trajectory history
        const auto& final_mix = *phd.extracted_mixtures().back();
        std::vector<double> steps_A, steps_B, hist_pos_A, hist_pos_B, hist_rot_A, hist_rot_B;

        for (std::size_t i = 0; i < final_mix.size(); ++i) {
            const auto& traj = final_mix.component(i);
            const auto& hist = traj.history();

            // Match this trajectory to the closest truth target at the final step
            Eigen::Vector3d final_pos = hist.back().mean().head(3);
            int best_t = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (int t = 0; t < 2; ++t) {
                double d = (final_pos - truth_targets[t].states.back().head(3)).norm();
                if (d < best_d) { best_d = d; best_t = t; }
            }

            // Walk through history and compute errors at each timestep
            const int h = static_cast<int>(hist.size());
            const int start_k = num_steps - h;
            auto& s_vec = (best_t == 0) ? steps_A : steps_B;
            auto& p_vec = (best_t == 0) ? hist_pos_A : hist_pos_B;
            auto& r_vec = (best_t == 0) ? hist_rot_A : hist_rot_B;

            for (int j = 0; j < h; ++j) {
                int k = start_k + j;
                if (k < 0 || k >= num_steps) continue;
                const auto& tgt = truth_targets[best_t];
                int idx = k - tgt.birth_time;
                if (idx < 0 || idx >= static_cast<int>(tgt.states.size())) continue;

                s_vec.push_back(static_cast<double>(k));
                p_vec.push_back((hist[j].mean().head(3) - tgt.states[idx].head(3)).norm());
                r_vec.push_back(rotation_error_3d(
                    Eigen::Matrix3d(hist[j].rotation()), tgt.rotations[idx]));
            }
        }

        auto ax1 = matplot::subplot(fig, 1, 2, 0);
        ax1->hold(true);
        if (!steps_A.empty()) {
            auto la = ax1->plot(steps_A, hist_pos_A); la->display_name("Plane"); la->line_width(2.0f);
        }
        if (!steps_B.empty()) {
            auto lb = ax1->plot(steps_B, hist_pos_B); lb->display_name("Jet"); lb->line_width(2.0f);
        }
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        if (!steps_A.empty()) {
            auto ra = ax2->plot(steps_A, hist_rot_A); ra->display_name("Plane"); ra->line_width(2.0f);
        }
        if (!steps_B.empty()) {
            auto rb = ax2->plot(steps_B, hist_rot_B); rb->display_name("Jet"); rb->line_width(2.0f);
        }
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/trajectory_tm_phd_3d_multi_convergence.png");
    }
#endif
}
