#include <gtest/gtest.h>
#include "brew/dynamics/single_integrator.hpp"
#include "brew/template_pose/template_pose_model.hpp"
#include "brew/shared/trajectory_window.hpp"
#include "brew/shared/mixture.hpp"
#include "brew/template_pose/filters/tm_ekf.hpp"
#include "brew/trajectory_template_pose/filters/trajectory_tm_ekf.hpp"
#include "brew/template_pose/template_pose.hpp"
#include "brew/trajectory_template_pose/trajectory_template_pose.hpp"
#include "brew/shared/fusion/prune.hpp"
#include "brew/clustering/dbscan.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include "brew/template_matching/point_to_point_icp.hpp"
#include "brew/template_matching/point_to_plane_icp.hpp"
#include "brew/template_matching/template_library.hpp"
#include "brew/desktop/measurement_sampling/measurement_sampling.hpp"

#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/desktop/plot_utils/plot_options.hpp>
#include <brew/desktop/plot_utils/color_palette.hpp>
#include <brew/desktop/plot_utils/plot_point_cloud.hpp>
#include <filesystem>
#endif

using namespace brew;

namespace {

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

}

namespace {

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

Eigen::Matrix3d facing_rotation(const Eigen::Vector3d& velocity) {
    Eigen::Vector3d fwd = velocity.normalized();

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

struct TmTruthTarget3D {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states;
    std::vector<Eigen::Matrix3d> rotations;
};

TmTruthTarget3D make_tm_target_3d(
    const Eigen::VectorXd& x0,
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& angular_vel,
    int birth, int death, double dt,
    dynamics::DynamicsBase<>& dyn)
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

        Eigen::Vector3d phi = angular_vel * dt;
        double angle = phi.norm();
        if (angle > 1e-10) {
            R = R * axis_angle_to_rotation(phi, angle);
        }
    }
    return t;
}

double rotation_error_3d(const Eigen::Matrix3d& R_est, const Eigen::Matrix3d& R_truth) {

    Eigen::Vector3d x_est = R_est.col(0);
    Eigen::Vector3d x_truth = R_truth.col(0);
    double cos_angle = std::clamp(x_est.dot(x_truth), -1.0, 1.0);
    return std::acos(cos_angle);
}

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

bool is_point_visible(
    const Eigen::Vector3d& world_pt,
    const Eigen::Vector3d& world_normal,
    const Eigen::Vector3d& sensor_pos,
    const Eigen::MatrixXd& world_tris,
    int num_tris,
    double cos_thresh = 0.0)
{
    Eigen::Vector3d to_sensor = sensor_pos - world_pt;
    double dist = to_sensor.norm();
    if (dist < 1e-10) return false;

    if (world_normal.dot(to_sensor) / dist < cos_thresh) return false;

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

Eigen::MatrixXd generate_tm_measurements_3d(
    const std::vector<TmTruthTarget3D>& targets,
    const std::vector<Eigen::MatrixXd>& mesh_triangles,
    const Eigen::Vector3d& sensor_pos,
    int timestep, double point_noise_std,
    std::mt19937& rng,
    int num_candidates = 1000,
    int num_meas_pts = 0,
    double cos_thresh = 0.0)
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

            std::vector<double> cum_area(num_tris);
            for (int t = 0; t < num_tris; ++t) {
                Eigen::Vector3d e1 = body_tris.col(3*t+1) - body_tris.col(3*t);
                Eigen::Vector3d e2 = body_tris.col(3*t+2) - body_tris.col(3*t);
                double area = 0.5 * e1.cross(e2).norm();
                cum_area[t] = (t > 0 ? cum_area[t-1] : 0.0) + area;
            }
            double total_area = cum_area.back();

            Eigen::MatrixXd world_tris = R * body_tris;
            world_tris.colwise() += pos;

            std::vector<Eigen::Vector3d> visible_pts;
            visible_pts.reserve(num_candidates);

            for (int c = 0; c < num_candidates; ++c) {

                double r = uniform(rng) * total_area;
                auto it = std::lower_bound(cum_area.begin(), cum_area.end(), r);
                int tri_idx = static_cast<int>(std::distance(cum_area.begin(), it));
                tri_idx = std::min(tri_idx, num_tris - 1);

                double u = uniform(rng), v = uniform(rng);
                double su = std::sqrt(u);
                double b0 = 1.0 - su, b1 = v * su, b2 = 1.0 - b0 - b1;
                Eigen::Vector3d body_pt = b0 * body_tris.col(3*tri_idx)
                                        + b1 * body_tris.col(3*tri_idx+1)
                                        + b2 * body_tris.col(3*tri_idx+2);

                Eigen::Vector3d e1 = body_tris.col(3*tri_idx+1) - body_tris.col(3*tri_idx);
                Eigen::Vector3d e2 = body_tris.col(3*tri_idx+2) - body_tris.col(3*tri_idx);
                Eigen::Vector3d n = e1.cross(e2);
                double len = n.norm();
                if (len < 1e-15) continue;
                n /= len;

                Eigen::Vector3d world_pt = R * body_pt + pos;
                Eigen::Vector3d world_normal = R * n;

                if (is_point_visible(world_pt, world_normal, sensor_pos,
                                     world_tris, num_tris, cos_thresh)) {
                    visible_pts.push_back(world_pt);
                }
            }

            if (visible_pts.empty()) continue;

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

}

TEST(TmPhd, Tracking3D_SingleTarget) {

    auto load_template = [](const std::string& stl_file, int n_pts) {
        auto cloud = measurement_sampling::sample_stl_vsp(stl_file, n_pts);
        Eigen::Vector3d centroid = cloud.points().rowwise().mean();
        cloud.points().colwise() -= centroid;
        auto templ = std::make_shared<template_matching::PointCloud>(cloud.points());
        return std::make_pair(templ, centroid);
    };

    auto [templ_B, centroid_B] = load_template("A.stl", 5000);

    auto lib = std::make_shared<template_matching::TemplateLibrary>();
    int id_B = lib->add(templ_B);

    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(3);

    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 25;
    icp_params.tolerance = 1e-10;
    icp_params.sigma_sq = 0.001;
    icp_params.trim_fraction = 1.0;

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    auto ekf = std::make_unique<filters::TmEkf<>>();
    ekf->set_dynamics(dyn);
    {
        Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
        ekf->set_process_noise(G * (0.50 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose());
    }
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.1 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);
    ekf->set_template_library(lib);

    const int num_steps = 10;
    const double dt = 0.2;
    const double point_noise_std = 0.0;

    Eigen::VectorXd x0(6);
    x0 << 0.0, 0.0, 50.0, 10.0, 5.0, 0.3;
    Eigen::Matrix3d R0 = facing_rotation(x0.tail(3));
    Eigen::Vector3d omega(0.0, 0.0, 0.0);

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0, R0, omega, 0, num_steps - 1, dt, *dyn));

    Eigen::MatrixXd tri_B_body = measurement_sampling::load_stl_triangles_vsp("A.stl");
    tri_B_body.colwise() -= centroid_B;

    std::vector<Eigen::MatrixXd> meas_triangles = {tri_B_body};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {

        Eigen::Vector3d sensor_pos(truth_targets[0].states[k](0),
                                   truth_targets[0].states[k](1), -1000.0);
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, meas_triangles, sensor_pos,
            k, point_noise_std, rng, 1000, 100, -0.3);
    }

    auto birth = std::make_unique<models::Mixture<models::TemplatePose<>>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 1500.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 500.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = (M_PI * M_PI) * Eigen::Matrix3d::Identity();
    std::vector<int> pos_indices = {0, 1, 2};

    Eigen::VectorXd b_mean(6);
    b_mean << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TemplatePose<>>(b_mean, birth_cov, id_B, pos_indices), 0.01);

    template_pose::PHD<> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::TemplatePose<>>>());
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

    std::cout << "\n=== TM-PHD 3D Single Target (B) ===\n";
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

    Eigen::MatrixXd tri_verts = measurement_sampling::load_stl_triangles_vsp("A.stl");
    tri_verts.colwise() -= centroid_B;
    const int num_tris = static_cast<int>(tri_verts.cols()) / 3;

    const int plot_stride = 3;

    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        for (int k = 0; k < num_steps; k += plot_stride) {

            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(5.0f);

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

                wx.push_back(std::nan("")); wy.push_back(std::nan("")); wz.push_back(std::nan(""));
            }
            auto wl = ax->plot3(wx, wy, wz);
            wl->color({0.f, 0.5f, 0.5f, 0.5f});
            wl->line_width(0.2f);
        }

        {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[0].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

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

    {
        auto fig = matplot::figure(true);
        fig->width(1200);
        fig->height(500);

        std::vector<double> steps(num_steps);
        for (int k = 0; k < num_steps; ++k) steps[k] = static_cast<double>(k);

        auto ax1 = matplot::subplot(fig, 1, 2, 0);
        ax1->hold(true);
        auto la = ax1->plot(steps, pos_errs); la->display_name("B"); la->line_width(2.0f);
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra = ax2->plot(steps, rot_errs); ra->display_name("B"); ra->line_width(2.0f);
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_single_convergence.png");
    }
#endif
}

TEST(TmPhd, Tracking3D_MultipleTargets) {

    struct TemplateData {
        std::shared_ptr<template_matching::PointCloud> templ;
        Eigen::Vector3d centroid;
        Eigen::Matrix3d pca_axes;
        Eigen::Vector3d pca_centroid;
    };
    auto load_template = [](const std::string& stl_file, int n_pts) {

        auto dense = measurement_sampling::sample_stl_vsp(stl_file, 5000);
        Eigen::Vector3d centroid = dense.points().rowwise().mean();
        dense.points().colwise() -= centroid;
        Eigen::Matrix3d axes = template_matching::PcaIcp::pca_axes(dense.points());
        Eigen::Vector3d pca_centroid = dense.points().rowwise().mean();

        auto cloud = measurement_sampling::sample_stl_vsp(stl_file, n_pts);
        cloud.points().colwise() -= centroid;
        auto templ = std::make_shared<template_matching::PointCloud>(cloud.points());
        return TemplateData{templ, centroid, axes, pca_centroid};
    };

    auto td_A = load_template("A.stl", 1000);
    auto td_B = load_template("B.stl", 1000);
    auto& templ_A = td_A.templ; auto& centroid_A = td_A.centroid;
    auto& templ_B = td_B.templ; auto& centroid_B = td_B.centroid;

    auto lib = std::make_shared<template_matching::TemplateLibrary>();
    int id_A = lib->add(templ_A, td_A.pca_axes, td_A.pca_centroid);
    int id_B = lib->add(templ_B, td_B.pca_axes, td_B.pca_centroid);

    std::cout << "\n--- Templates (VSP-aligned) ---\n";
    std::cout << "A: " << templ_A->num_points() << " pts, bbox X["
              << templ_A->points().row(0).minCoeff() << ", " << templ_A->points().row(0).maxCoeff()
              << "] Y[" << templ_A->points().row(1).minCoeff() << ", " << templ_A->points().row(1).maxCoeff()
              << "] Z[" << templ_A->points().row(2).minCoeff() << ", " << templ_A->points().row(2).maxCoeff() << "]\n";
    std::cout << "B:   " << templ_B->num_points() << " pts, bbox X["
              << templ_B->points().row(0).minCoeff() << ", " << templ_B->points().row(0).maxCoeff()
              << "] Y[" << templ_B->points().row(1).minCoeff() << ", " << templ_B->points().row(1).maxCoeff()
              << "] Z[" << templ_B->points().row(2).minCoeff() << ", " << templ_B->points().row(2).maxCoeff() << "]\n";

    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(3);

    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 25;
    icp_params.tolerance = 1e-12;
    icp_params.sigma_sq = 0.1;

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    auto ekf = std::make_unique<filters::TmEkf<>>();
    ekf->set_dynamics(dyn);
    {
        Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
        ekf->set_process_noise(G * (0.5 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose());
    }
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.1 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.05 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);
    ekf->set_template_library(lib);

    const int num_steps = 25;
    const double dt = 0.2;
    const double point_noise_std = 0.0;
    const double p_detect = 0.95;

    Eigen::VectorXd x0_A(6);
    x0_A << 0.0, 0.0, 50.0, 15.0, 10.0, 0.5;
    Eigen::Matrix3d R0_A = facing_rotation(x0_A.tail(3));
    Eigen::Vector3d omega_A(0.0, 0.0, 0.0);

    Eigen::VectorXd x0_B(6);
    x0_B << 50.0, 80.0, 80.0, 20.0, -5.0, -0.3;
    Eigen::Matrix3d R0_B = facing_rotation(x0_B.tail(3));
    Eigen::Vector3d omega_B(0.0, 0.0, 0.0);

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0_A, R0_A, omega_A, 0, num_steps - 1, dt, *dyn));
    truth_targets.push_back(make_tm_target_3d(x0_B,   R0_B,   omega_B,   0, num_steps - 1, dt, *dyn));

    Eigen::MatrixXd tri_A_body = measurement_sampling::load_stl_triangles_vsp("A.stl");
    tri_A_body.colwise() -= centroid_A;

    Eigen::MatrixXd tri_B_body = measurement_sampling::load_stl_triangles_vsp("B.stl");
    tri_B_body.colwise() -= centroid_B;

    std::vector<Eigen::MatrixXd> meas_triangles = {tri_A_body, tri_B_body};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {

        Eigen::Vector3d mid = Eigen::Vector3d::Zero();
        int n_active = 0;
        for (const auto& tgt : truth_targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                mid += tgt.states[k - tgt.birth_time].head(3);
                ++n_active;
            }
        }
        if (n_active > 0) mid /= static_cast<double>(n_active);
        Eigen::Vector3d sensor_pos(0.0, 0.0, 0.0);
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, meas_triangles, sensor_pos,
            k, point_noise_std, rng, 1000, 100, -0.3);
    }

    auto birth = std::make_unique<models::Mixture<models::TemplatePose<>>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 1500.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 500.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = 1.0 * Eigen::Matrix3d::Identity();
    std::vector<int> pos_indices = {0, 1, 2};

    Eigen::VectorXd b_mean(6);
    b_mean << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    double w_b = 0.001;
    birth->add_component(
        std::make_unique<models::TemplatePose<>>(b_mean, birth_cov, id_A, pos_indices), w_b);
    birth->add_component(
        std::make_unique<models::TemplatePose<>>(b_mean, birth_cov, id_B, pos_indices), w_b);

    template_pose::PHD<> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::TemplatePose<>>>());
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

    std::cout << "\n=== TM-PHD 3D Multi-Template Tracking (A + B) ===\n";
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

    {

        Eigen::MatrixXd tri_A_body = measurement_sampling::load_stl_triangles_vsp("A.stl");
        tri_A_body.colwise() -= centroid_A;
        const int n_tri_A_body = static_cast<int>(tri_A_body.cols()) / 3;

        Eigen::MatrixXd tri_B_body = measurement_sampling::load_stl_triangles_vsp("B.stl");
        tri_B_body.colwise() -= centroid_B;
        const int n_tri_B_body = static_cast<int>(tri_B_body.cols()) / 3;

        auto render_mesh = [](matplot::axes_handle ax,
                              const Eigen::MatrixXd& tris, int n_tris,
                              const std::array<float, 4>& color) {
            for (int t = 0; t < n_tris; ++t) {
                std::vector<double> xs = {tris(0, 3*t), tris(0, 3*t+1), tris(0, 3*t+2), tris(0, 3*t)};
                std::vector<double> ys = {tris(1, 3*t), tris(1, 3*t+1), tris(1, 3*t+2), tris(1, 3*t)};
                std::vector<double> zs = {tris(2, 3*t), tris(2, 3*t+1), tris(2, 3*t+2), tris(2, 3*t)};
                auto l = ax->plot3(xs, ys, zs, "-");
                l->color(color);
                l->line_width(0.5f);
            }
        };

        auto set_equal_axes = [](matplot::axes_handle ax, const Eigen::MatrixXd& tris) {
            double xmin = tris.row(0).minCoeff(), xmax = tris.row(0).maxCoeff();
            double ymin = tris.row(1).minCoeff(), ymax = tris.row(1).maxCoeff();
            double zmin = tris.row(2).minCoeff(), zmax = tris.row(2).maxCoeff();
            double cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2, cz = (zmin + zmax) / 2;
            double half = std::max({xmax - xmin, ymax - ymin, zmax - zmin}) / 2 * 1.1;
            ax->xlim({cx - half, cx + half});
            ax->ylim({cy - half, cy + half});
            ax->zlim({cz - half, cz + half});
        };

        {
            auto fig = matplot::figure(true);
            fig->width(800);
            fig->height(600);
            auto ax = fig->current_axes();
            ax->hold(true);
            render_mesh(ax, tri_A_body, n_tri_A_body, {0.2f, 0.4f, 0.8f, 1.0f});
            ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
            ax->x_axis().label_font_size(18);
            ax->y_axis().label_font_size(18);
            ax->z_axis().label_font_size(18);
            set_equal_axes(ax, tri_A_body);
            ax->view(45.f, 30.f);
            brew::plot_utils::save_figure(fig, "output/stl_A.png");
        }

        {
            auto fig = matplot::figure(true);
            fig->width(800);
            fig->height(600);
            auto ax = fig->current_axes();
            ax->hold(true);
            render_mesh(ax, tri_B_body, n_tri_B_body, {0.8f, 0.2f, 0.2f, 1.0f});
            ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
            ax->x_axis().label_font_size(18);
            ax->y_axis().label_font_size(18);
            ax->z_axis().label_font_size(18);
            set_equal_axes(ax, tri_B_body);
            ax->view(45.f, 30.f);
            brew::plot_utils::save_figure(fig, "output/stl_B.png");
        }
    }

    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        const int plot_stride = 3;

        Eigen::MatrixXd tri_A = measurement_sampling::load_stl_triangles_vsp("A.stl");
        tri_A.colwise() -= centroid_A;

        const int n_tri_A = static_cast<int>(tri_A.cols()) / 3;

        Eigen::MatrixXd tri_B = measurement_sampling::load_stl_triangles_vsp("B.stl");
        tri_B.colwise() -= centroid_B;

        const int n_tri_B = static_cast<int>(tri_B.cols()) / 3;

        auto get_tri_data = [&](const models::TemplatePose<>& est)
            -> std::pair<const Eigen::MatrixXd*, int> {
            if (est.template_id() == id_A)
                return {&tri_A, n_tri_A};
            return {&tri_B, n_tri_B};
        };

        for (int k = 0; k < num_steps; k += plot_stride) {

            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(3.0f);

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

        for (int t = 0; t < 2; ++t) {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[t].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

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

    {
        auto fig = matplot::figure(true);
        fig->width(1200);
        fig->height(500);

        std::vector<double> steps(num_steps);
        for (int k = 0; k < num_steps; ++k) steps[k] = static_cast<double>(k);

        auto ax1 = matplot::subplot(fig, 1, 2, 0);
        ax1->hold(true);
        auto la = ax1->plot(steps, pos_A); la->display_name("A"); la->line_width(2.0f);
        auto lb = ax1->plot(steps, pos_B); lb->display_name("B"); lb->line_width(2.0f);
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        auto ra = ax2->plot(steps, rot_A); ra->display_name("A"); ra->line_width(2.0f);
        auto rb = ax2->plot(steps, rot_B); rb->display_name("B"); rb->line_width(2.0f);
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/tm_phd_3d_multi_convergence.png");
    }

#endif
}

TEST(TmPhd, TrajectoryTracking3D_MultipleTargets) {

    struct TemplateData {
        std::shared_ptr<template_matching::PointCloud> templ;
        Eigen::Vector3d centroid;
        Eigen::Matrix3d pca_axes;
        Eigen::Vector3d pca_centroid;
    };
    auto load_template = [](const std::string& stl_file, int n_pts) {
        auto dense = measurement_sampling::sample_stl_vsp(stl_file, 5000);
        Eigen::Vector3d centroid = dense.points().rowwise().mean();
        dense.points().colwise() -= centroid;
        Eigen::Matrix3d axes = template_matching::PcaIcp::pca_axes(dense.points());
        Eigen::Vector3d pca_centroid = dense.points().rowwise().mean();

        auto cloud = measurement_sampling::sample_stl_vsp(stl_file, n_pts);
        cloud.points().colwise() -= centroid;
        auto templ = std::make_shared<template_matching::PointCloud>(cloud.points());
        return TemplateData{templ, centroid, axes, pca_centroid};
    };

    auto td_A = load_template("A.stl", 1000);
    auto td_B = load_template("B.stl", 1000);
    auto& templ_A = td_A.templ; auto& centroid_A = td_A.centroid;
    auto& templ_B = td_B.templ; auto& centroid_B = td_B.centroid;

    auto lib = std::make_shared<template_matching::TemplateLibrary>();
    int id_A = lib->add(templ_A, td_A.pca_axes, td_A.pca_centroid);
    int id_B = lib->add(templ_B, td_B.pca_axes, td_B.pca_centroid);

    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(3);

    template_matching::IcpParams icp_params;
    icp_params.max_iterations = 25;
    icp_params.tolerance = 1e-6;
    icp_params.sigma_sq = 0.1;
    icp_params.trim_fraction = 1.0;

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    constexpr int kTrajTmWindow3D = 5;
    auto ekf = std::make_unique<filters::TrajectoryTmEkf<>>();
    ekf->set_dynamics(dyn);
    {
        Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
        ekf->set_process_noise(G * (0.5 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose());
    }
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = 0.01 * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(0.05 * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);
    ekf->set_template_library(lib);

    const int num_steps = 25;
    const double dt = 0.2;
    const double point_noise_std = 0.0;
    const double p_detect = 0.95;

    Eigen::VectorXd x0_A(6);
    x0_A << 0.0, 0.0, 50.0, 15.0, 10.0, 0.5;
    Eigen::Matrix3d R0_A = facing_rotation(x0_A.tail(3));
    Eigen::Vector3d omega_A(0.0, 0.0, 0.0);

    Eigen::VectorXd x0_B(6);
    x0_B << 50.0, 80.0, 80.0, 20.0, -5.0, -0.3;
    Eigen::Matrix3d R0_B = facing_rotation(x0_B.tail(3));
    Eigen::Vector3d omega_B(0.0, 0.0, 0.0);

    std::vector<TmTruthTarget3D> truth_targets;
    truth_targets.push_back(make_tm_target_3d(x0_A, R0_A, omega_A, 0, num_steps - 1, dt, *dyn));
    truth_targets.push_back(make_tm_target_3d(x0_B,   R0_B,   omega_B,   0, num_steps - 1, dt, *dyn));

    Eigen::MatrixXd tri_A_body = measurement_sampling::load_stl_triangles_vsp("A.stl");
    tri_A_body.colwise() -= centroid_A;

    Eigen::MatrixXd tri_B_body = measurement_sampling::load_stl_triangles_vsp("B.stl");
    tri_B_body.colwise() -= centroid_B;

    std::vector<Eigen::MatrixXd> meas_triangles = {tri_A_body, tri_B_body};

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {

        Eigen::Vector3d mid = Eigen::Vector3d::Zero();
        int n_active = 0;
        for (const auto& tgt : truth_targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                mid += tgt.states[k - tgt.birth_time].head(3);
                ++n_active;
            }
        }
        if (n_active > 0) mid /= static_cast<double>(n_active);
        Eigen::Vector3d sensor_pos(0.0, 0.0, 0.0);
        measurements[k] = generate_tm_measurements_3d(
            truth_targets, meas_triangles, sensor_pos,
            k, point_noise_std, rng, 1000, 100, -0.3);
    }

    auto birth = std::make_unique<models::Mixture<models::TrajectoryTemplatePose<>>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3) = 1500.0 * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3) = 500.0 * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = 1.0 * Eigen::Matrix3d::Identity();
    std::vector<int> pos_indices = {0, 1, 2};
    const int sd = 6;

    Eigen::VectorXd b_mean(sd);
    b_mean << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    auto make_traj_birth = [&](int template_id) {

        auto tp = models::TemplatePose<>(b_mean, birth_cov, template_id, pos_indices);

        models::TemplatePose<> tp_trans = tp;
        tp_trans.covariance() = tp.covariance().topLeftCorner(sd, sd);
        auto traj = std::make_unique<
            models::TrajectoryTemplatePose<>>(sd, tp_trans, kTrajTmWindow3D);
        traj->history_at(0) = tp;
        return traj;
    };

    double w_b = 0.001;

    birth->add_component(make_traj_birth(id_A), w_b);
    birth->add_component(make_traj_birth(id_B), w_b);

    trajectory_template_pose::PHD<> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(
        std::make_unique<models::Mixture<models::TrajectoryTemplatePose<>>>());
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

    std::cout << "\n=== Trajectory TM-PHD 3D Multi-Template Tracking (A + B) ===\n";
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

    {
        auto fig = matplot::figure(true);
        fig->width(1100);
        fig->height(900);
        auto ax = fig->current_axes();
        ax->hold(true);

        const int plot_stride = 3;

        Eigen::MatrixXd tri_A = measurement_sampling::load_stl_triangles_vsp("A.stl");
        tri_A.colwise() -= centroid_A;

        const int n_tri_A = static_cast<int>(tri_A.cols()) / 3;

        Eigen::MatrixXd tri_B = measurement_sampling::load_stl_triangles_vsp("B.stl");
        tri_B.colwise() -= centroid_B;

        const int n_tri_B = static_cast<int>(tri_B.cols()) / 3;

        auto get_tri_data = [&](const models::TemplatePose<>& est)
            -> std::pair<const Eigen::MatrixXd*, int> {
            if (est.template_id() == id_A)
                return {&tri_A, n_tri_A};
            return {&tri_B, n_tri_B};
        };

        for (int k = 0; k < num_steps; k += plot_stride) {

            const auto& m = measurements[k];
            std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
            for (int j = 0; j < m.cols(); ++j) {
                mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
            }
            auto mp = ax->plot3(mx, my, mz, ".");
            mp->color({0.f, 1.0f, 0.0f, 0.0f});
            mp->marker_size(3.0f);

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

        for (int t = 0; t < 2; ++t) {
            std::vector<double> tx, ty, tz;
            for (const auto& s : truth_targets[t].states) {
                tx.push_back(s(0)); ty.push_back(s(1)); tz.push_back(s(2));
            }
            auto tl = ax->plot3(tx, ty, tz);
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(2.0f);
        }

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

    {
        auto fig = matplot::figure(true);
        fig->width(1200);
        fig->height(500);

        const auto& final_mix = *phd.extracted_mixtures().back();
        std::vector<double> steps_A, steps_B, hist_pos_A, hist_pos_B, hist_rot_A, hist_rot_B;

        for (std::size_t i = 0; i < final_mix.size(); ++i) {
            const auto& traj = final_mix.component(i);
            const auto& hist = traj.history();

            Eigen::Vector3d final_pos = hist.back().mean().head(3);
            int best_t = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (int t = 0; t < 2; ++t) {
                double d = (final_pos - truth_targets[t].states.back().head(3)).norm();
                if (d < best_d) { best_d = d; best_t = t; }
            }

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
            auto la = ax1->plot(steps_A, hist_pos_A); la->display_name("A"); la->line_width(2.0f);
        }
        if (!steps_B.empty()) {
            auto lb = ax1->plot(steps_B, hist_pos_B); lb->display_name("B"); lb->line_width(2.0f);
        }
        ax1->xlabel("(a)");
        ax1->x_axis().label_font_size(18);

        auto ax2 = matplot::subplot(fig, 1, 2, 1);
        ax2->hold(true);
        if (!steps_A.empty()) {
            auto ra = ax2->plot(steps_A, hist_rot_A); ra->display_name("A"); ra->line_width(2.0f);
        }
        if (!steps_B.empty()) {
            auto rb = ax2->plot(steps_B, hist_rot_B); rb->display_name("B"); rb->line_width(2.0f);
        }
        ax2->xlabel("(b)");
        ax2->x_axis().label_font_size(18);

        brew::plot_utils::save_figure(fig, "output/trajectory_tm_phd_3d_multi_convergence.png");
    }
#endif
}
