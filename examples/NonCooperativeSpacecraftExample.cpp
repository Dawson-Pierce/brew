// ============================================================
// Non-Cooperative Spacecraft TM-PHD Tracking Example
// ============================================================
// Demonstrates RFS-based multi-target tracking with template
// matching and a bounding-box sensor field of view.  All tunable
// parameters are loaded from a YAML config file so the example
// can be re-run without recompiling.
// ============================================================

#include "brew/core/dynamics/single_integrator.hpp"
#include "brew/core/models/template_pose.hpp"
#include "brew/core/models/mixture.hpp"
#include "brew/core/filters/tm_ekf.hpp"
#include "brew/advanced/multi_target/phd.hpp"
#include "brew/core/fusion/prune.hpp"
#include "brew/core/fusion/merge.hpp"
#include "brew/advanced/clustering/dbscan.hpp"
#include "brew/core/template_matching/point_cloud.hpp"
#include "brew/core/template_matching/point_to_plane_icp.hpp"
#include "brew/core/template_matching/pca_icp.hpp"
#include "brew/desktop/measurement_sampling/measurement_sampling.hpp"

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <filesystem>
#include <string>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/desktop/plot_utils/plot_options.hpp>
#include <brew/desktop/plot_utils/color_palette.hpp>
#include <brew/desktop/plot_utils/plot_point_cloud.hpp>
#endif

using namespace brew;

// ---- Geometry helpers -------------------------------------------------------

static Eigen::Matrix3d axis_angle_to_rotation(const Eigen::Vector3d& axis,
                                               double angle) {
    Eigen::Vector3d a = axis.normalized();
    Eigen::Matrix3d K;
    K <<  0.0, -a(2),  a(1),
          a(2),  0.0, -a(0),
         -a(1),  a(0),  0.0;
    return Eigen::Matrix3d::Identity()
           + std::sin(angle) * K
           + (1.0 - std::cos(angle)) * K * K;
}

static Eigen::Matrix3d facing_rotation(const Eigen::Vector3d& velocity) {
    Eigen::Vector3d fwd = velocity.normalized();
    Eigen::Vector3d up_hint = Eigen::Vector3d::UnitZ();
    if (std::abs(fwd.dot(up_hint)) > 0.99)
        up_hint = Eigen::Vector3d::UnitY();
    Eigen::Vector3d right = up_hint.cross(fwd).normalized();
    Eigen::Vector3d up    = fwd.cross(right).normalized();
    Eigen::Matrix3d R;
    R.col(0) = fwd;
    R.col(1) = right;
    R.col(2) = up;
    return R;
}

static Eigen::MatrixXd transform_3d(const Eigen::MatrixXd& pts,
                                     const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& t) {
    Eigen::MatrixXd out = R * pts;
    out.colwise() += t;
    return out;
}

// ---- Visibility / ray tracing -----------------------------------------------

static bool ray_triangle_intersect(
    const Eigen::Vector3d& origin, const Eigen::Vector3d& dir,
    const Eigen::Vector3d& v0, const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2, double max_t, double eps = 1e-6)
{
    Eigen::Vector3d e1 = v1 - v0, e2 = v2 - v0;
    Eigen::Vector3d h = dir.cross(e2);
    double a = e1.dot(h);
    if (std::abs(a) < eps) return false;
    double f = 1.0 / a;
    Eigen::Vector3d s = origin - v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;
    Eigen::Vector3d q = s.cross(e1);
    double v = f * dir.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;
    double t = f * e2.dot(q);
    return t > eps && t < max_t - eps;
}

static bool is_point_visible(
    const Eigen::Vector3d& pt, const Eigen::Vector3d& normal,
    const Eigen::Vector3d& sensor_pos,
    const Eigen::MatrixXd& world_tris, int num_tris,
    double cos_thresh)
{
    Eigen::Vector3d to_sensor = sensor_pos - pt;
    double dist = to_sensor.norm();
    if (dist < 1e-10) return false;
    if (normal.dot(to_sensor) / dist < cos_thresh) return false;

    Eigen::Vector3d origin = pt + 0.01 * normal;
    Eigen::Vector3d dir    = (sensor_pos - origin).normalized();
    double ray_dist = (sensor_pos - origin).norm();
    for (int t = 0; t < num_tris; ++t) {
        if (ray_triangle_intersect(origin, dir,
                world_tris.col(3*t), world_tris.col(3*t+1), world_tris.col(3*t+2),
                ray_dist))
            return false;
    }
    return true;
}

// ---- Truth target -----------------------------------------------------------

struct TruthTarget3D {
    int birth_time, death_time;
    int template_index;
    std::vector<Eigen::VectorXd>  states;     // [x,y,z,vx,vy,vz]
    std::vector<Eigen::Matrix3d>  rotations;
};

static TruthTarget3D make_truth_target(
    const Eigen::VectorXd& x0, const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& omega, int birth, int death, double dt,
    int template_index, dynamics::DynamicsBase<>& dyn)
{
    TruthTarget3D tgt;
    tgt.birth_time = birth;
    tgt.death_time = death;
    tgt.template_index = template_index;
    Eigen::VectorXd x = x0;
    Eigen::Matrix3d R = R0;
    for (int k = birth; k <= death; ++k) {
        tgt.states.push_back(x);
        tgt.rotations.push_back(R);
        x = dyn.propagate_state(dt, x);
        Eigen::Vector3d phi = omega * dt;
        double angle = phi.norm();
        if (angle > 1e-10)
            R = R * axis_angle_to_rotation(phi, angle);
    }
    return tgt;
}

// ---- Measurement generation with bounding-box sensor FOV -------------------

static Eigen::MatrixXd generate_measurements(
    const std::vector<TruthTarget3D>& targets,
    const std::vector<Eigen::MatrixXd>& mesh_triangles,
    const Eigen::Vector3d& sensor_pos,
    const Eigen::Vector3d& bbox_min,
    const Eigen::Vector3d& bbox_max,
    int timestep, double noise_std,
    std::mt19937& rng,
    int num_candidates, int num_meas_pts, double cos_thresh)
{
    std::normal_distribution<double> noise(0.0, noise_std);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Eigen::MatrixXd> clouds;

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        const auto& tgt = targets[ti];
        if (timestep < tgt.birth_time || timestep > tgt.death_time) continue;

        int idx = timestep - tgt.birth_time;
        Eigen::Vector3d pos = tgt.states[idx].head(3);
        const Eigen::Matrix3d& R = tgt.rotations[idx];

        const Eigen::MatrixXd& body_tris = mesh_triangles[ti];
        int num_tris = static_cast<int>(body_tris.cols()) / 3;

        // Cumulative area for face-weighted sampling
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

        std::vector<Eigen::Vector3d> visible;
        visible.reserve(num_candidates);

        for (int c = 0; c < num_candidates; ++c) {
            double r = uniform(rng) * total_area;
            auto it = std::lower_bound(cum_area.begin(), cum_area.end(), r);
            int tri_idx = std::min(static_cast<int>(std::distance(cum_area.begin(), it)),
                                   num_tris - 1);

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

            Eigen::Vector3d world_pt     = R * body_pt + pos;
            Eigen::Vector3d world_normal = R * n;

            if (!is_point_visible(world_pt, world_normal, sensor_pos,
                                  world_tris, num_tris, cos_thresh))
                continue;

            // Bounding-box sensor FOV mask
            if (world_pt(0) < bbox_min(0) || world_pt(0) > bbox_max(0) ||
                world_pt(1) < bbox_min(1) || world_pt(1) > bbox_max(1) ||
                world_pt(2) < bbox_min(2) || world_pt(2) > bbox_max(2))
                continue;

            visible.push_back(world_pt);
        }

        if (visible.empty()) continue;

        if (num_meas_pts > 0 && num_meas_pts < static_cast<int>(visible.size())) {
            std::shuffle(visible.begin(), visible.end(), rng);
            visible.resize(num_meas_pts);
        }

        Eigen::MatrixXd pts(3, static_cast<int>(visible.size()));
        for (int j = 0; j < static_cast<int>(visible.size()); ++j) {
            pts.col(j) = visible[j];
            pts(0, j) += noise(rng);
            pts(1, j) += noise(rng);
            pts(2, j) += noise(rng);
        }
        clouds.push_back(pts);
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

// ---- Rotation error ---------------------------------------------------------

static double rotation_error_3d(const Eigen::Matrix3d& R_est,
                                const Eigen::Matrix3d& R_truth) {
    Eigen::Vector3d x_est   = R_est.col(0);
    Eigen::Vector3d x_truth = R_truth.col(0);
    return std::acos(std::clamp(x_est.dot(x_truth), -1.0, 1.0));
}

// ---- YAML helpers -----------------------------------------------------------

static Eigen::Vector3d yaml_to_vec3(const YAML::Node& node) {
    return Eigen::Vector3d(node[0].as<double>(),
                           node[1].as<double>(),
                           node[2].as<double>());
}

static Eigen::VectorXd yaml_to_vec(const YAML::Node& node, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = node[i].as<double>();
    return v;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    // --- Load config ---------------------------------------------------------
    std::string config_path = "examples/NonCooperativeSpacecraftConfig.yaml";
    if (argc > 1) config_path = argv[1];

    std::cout << "Loading config: " << config_path << "\n";
    YAML::Node cfg = YAML::LoadFile(config_path);

    const auto& sim_cfg   = cfg["simulation"];
    const auto& sens_cfg  = cfg["sensor"];
    const auto& icp_cfg   = cfg["icp"];
    const auto& filt_cfg  = cfg["filter"];
    const auto& phd_cfg   = cfg["phd"];
    const auto& birth_cfg = cfg["birth"];
    const auto& clust_cfg = cfg["clustering"];
    const auto& vis_cfg   = cfg["visualization"];

    const int    num_steps       = sim_cfg["num_steps"].as<int>();
    const double dt              = sim_cfg["dt"].as<double>();
    const int    seed            = sim_cfg["seed"].as<int>();
    const double point_noise_std = sim_cfg["point_noise_std"].as<double>();

    const Eigen::Vector3d bbox_min = yaml_to_vec3(sens_cfg["bbox_min"]);
    const Eigen::Vector3d bbox_max = yaml_to_vec3(sens_cfg["bbox_max"]);
    const int    meas_candidates   = sens_cfg["measurement_candidates"].as<int>();
    const int    meas_points       = sens_cfg["measurement_points"].as<int>();
    const double cos_vis_thresh    = sens_cfg["cos_visibility_threshold"].as<double>();

    const std::string output_dir = vis_cfg["output_dir"].as<std::string>();

    // --- Load templates ------------------------------------------------------
    struct TemplateData {
        std::shared_ptr<template_matching::PointCloud> templ;
        Eigen::Vector3d centroid;
        Eigen::Matrix3d pca_axes;
        Eigen::Vector3d pca_centroid;
        Eigen::MatrixXd body_triangles;   // centered body-frame tris
    };

    std::vector<TemplateData> templates;
    for (const auto& t_cfg : cfg["templates"]) {
        std::string file     = t_cfg["file"].as<std::string>();
        int n_templ          = t_cfg["num_template_points"].as<int>();
        int n_dense          = t_cfg["num_dense_points"].as<int>();

        auto dense = measurement_sampling::sample_stl_vsp(file, n_dense);
        Eigen::Vector3d centroid = dense.points().rowwise().mean();
        dense.points().colwise() -= centroid;
        Eigen::Matrix3d axes = template_matching::PcaIcp::pca_axes(dense.points());
        Eigen::Vector3d pca_centroid = dense.points().rowwise().mean();

        auto cloud = measurement_sampling::sample_stl_vsp(file, n_templ);
        cloud.points().colwise() -= centroid;
        auto templ = std::make_shared<template_matching::PointCloud>(cloud.points());

        Eigen::MatrixXd tris = measurement_sampling::load_stl_triangles_vsp(file);
        tris.colwise() -= centroid;

        templates.push_back({templ, centroid, axes, pca_centroid, tris});
        std::cout << "Template " << file << ": " << templ->num_points() << " pts\n";
    }

    // --- Dynamics ------------------------------------------------------------
    auto dyn = std::make_shared<dynamics::SingleIntegrator<>>(3);

    // --- ICP -----------------------------------------------------------------
    template_matching::IcpParams icp_params;
    icp_params.max_iterations = icp_cfg["max_iterations"].as<int>();
    icp_params.tolerance      = icp_cfg["tolerance"].as<double>();
    icp_params.sigma_sq       = icp_cfg["sigma_sq"].as<double>();

    auto inner_icp = std::make_shared<template_matching::PointToPlaneIcp>();
    inner_icp->set_params(icp_params);

    // --- TM-EKF --------------------------------------------------------------
    auto ekf = std::make_unique<filters::TmEkf<>>();
    ekf->set_dynamics(dyn);
    ekf->set_process_noise(
        filt_cfg["process_noise"].as<double>() * Eigen::MatrixXd::Identity(3, 3));
    Eigen::MatrixXd R_meas = Eigen::MatrixXd::Zero(6, 6);
    R_meas.topLeftCorner(3, 3)     = filt_cfg["measurement_noise_pos"].as<double>()
                                     * Eigen::Matrix3d::Identity();
    R_meas.bottomRightCorner(3, 3) = filt_cfg["measurement_noise_rot"].as<double>()
                                     * Eigen::Matrix3d::Identity();
    ekf->set_measurement_noise(R_meas);
    ekf->set_rotation_process_noise(
        filt_cfg["rotation_process_noise"].as<double>() * Eigen::MatrixXd::Identity(3, 3));
    ekf->set_icp(inner_icp);

    for (auto& td : templates)
        ekf->set_pca_prior(td.templ.get(), td.pca_axes, td.pca_centroid);

    // --- Truth targets -------------------------------------------------------
    std::vector<TruthTarget3D> truth_targets;
    std::vector<Eigen::MatrixXd> target_mesh_tris;   // per-target body tris

    for (const auto& tgt_cfg : cfg["targets"]) {
        int ti    = tgt_cfg["template_index"].as<int>();
        int birth = tgt_cfg["birth_time"].as<int>();
        int death = tgt_cfg["death_time"].as<int>();
        Eigen::VectorXd x0    = yaml_to_vec(tgt_cfg["initial_state"], 6);
        Eigen::Vector3d omega = yaml_to_vec3(tgt_cfg["angular_velocity"]);

        Eigen::Matrix3d R0 = facing_rotation(x0.tail(3));
        truth_targets.push_back(
            make_truth_target(x0, R0, omega, birth, death, dt, ti, *dyn));
        target_mesh_tris.push_back(templates[ti].body_triangles);
    }

    // --- Generate measurements -----------------------------------------------
    std::mt19937 rng(seed);

    // Sensor at origin (observing spacecraft from ground/another spacecraft)
    Eigen::Vector3d sensor_pos(0.0, 0.0, 0.0);

    std::vector<Eigen::MatrixXd> measurements(num_steps);
    for (int k = 0; k < num_steps; ++k) {
        measurements[k] = generate_measurements(
            truth_targets, target_mesh_tris, sensor_pos,
            bbox_min, bbox_max,
            k, point_noise_std, rng,
            meas_candidates, meas_points, cos_vis_thresh);
    }

    // --- Birth model ---------------------------------------------------------
    auto birth = std::make_unique<models::Mixture<models::TemplatePose<>>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Zero(9, 9);
    birth_cov.topLeftCorner(3, 3)     = birth_cfg["cov_position"].as<double>()
                                        * Eigen::Matrix3d::Identity();
    birth_cov.block(3, 3, 3, 3)      = birth_cfg["cov_velocity"].as<double>()
                                        * Eigen::Matrix3d::Identity();
    birth_cov.bottomRightCorner(3, 3) = birth_cfg["cov_rotation"].as<double>()
                                        * Eigen::Matrix3d::Identity();
    Eigen::MatrixXd R_birth;   // empty -> PCA-ICP cold start
    std::vector<int> pos_indices = {0, 1, 2};
    Eigen::VectorXd b_mean = yaml_to_vec(birth_cfg["mean"], 6);
    double w_b = birth_cfg["weight"].as<double>();

    for (auto& td : templates) {
        birth->add_component(
            std::make_unique<models::TemplatePose<>>(
                b_mean, birth_cov, R_birth, td.templ, pos_indices),
            w_b);
    }

    // --- PHD filter ----------------------------------------------------------
    multi_target::PHD<models::TemplatePose<>> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<models::TemplatePose<>>>());
    phd.set_prob_detection(phd_cfg["prob_detection"].as<double>());
    phd.set_prob_survive(phd_cfg["prob_survive"].as<double>());
    phd.set_clutter_rate(phd_cfg["clutter_rate"].as<double>());
    phd.set_clutter_density(phd_cfg["clutter_density"].as<double>());
    phd.set_prune_threshold(phd_cfg["prune_threshold"].as<double>());
    phd.set_merge_threshold(phd_cfg["merge_threshold"].as<double>());
    phd.set_max_components(phd_cfg["max_components"].as<int>());
    phd.set_extract_threshold(phd_cfg["extract_threshold"].as<double>());
    phd.set_gate_threshold(phd_cfg["gate_threshold"].as<double>());
    phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(
        clust_cfg["epsilon"].as<double>(),
        clust_cfg["min_points"].as<int>()));

    // --- Run filter ----------------------------------------------------------
    std::cout << "\n=== Non-Cooperative Spacecraft TM-PHD ===\n";
    std::cout << "Sensor bbox: [" << bbox_min.transpose() << "] -> ["
              << bbox_max.transpose() << "]\n\n";

    std::cout << std::setw(5)  << "k"
              << std::setw(10) << "n_meas"
              << std::setw(10) << "n_comp"
              << std::setw(10) << "n_est";
    for (std::size_t t = 0; t < truth_targets.size(); ++t)
        std::cout << std::setw(10) << ("pos_" + std::to_string(t))
                  << std::setw(10) << ("rot_" + std::to_string(t));
    std::cout << "\n";

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);
        phd.correct(measurements[k]);
        phd.cleanup();

        const auto& latest = *phd.extracted_mixtures().back();

        std::cout << std::setw(5)  << k
                  << std::setw(10) << measurements[k].cols()
                  << std::setw(10) << phd.intensity().size()
                  << std::setw(10) << latest.size();

        for (std::size_t t = 0; t < truth_targets.size(); ++t) {
            const auto& tgt = truth_targets[t];
            double best_pos = std::numeric_limits<double>::infinity();
            double best_rot = std::numeric_limits<double>::infinity();
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::Vector3d truth_pos = tgt.states[idx].head(3);
                const Eigen::Matrix3d& truth_R = tgt.rotations[idx];
                for (std::size_t i = 0; i < latest.size(); ++i) {
                    double pe = (latest.component(i).mean().head(3) - truth_pos).norm();
                    if (pe < best_pos) {
                        best_pos = pe;
                        best_rot = rotation_error_3d(
                            Eigen::Matrix3d(latest.component(i).rotation()), truth_R);
                    }
                }
            }
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << best_pos
                      << std::setw(10) << best_rot;
        }
        std::cout << "\n";

        // --- Per-timestep visualization --------------------------------------
#ifdef BREW_ENABLE_PLOTTING
        {
            std::filesystem::create_directories(output_dir);

            const int fig_w   = vis_cfg["fig_width"].as<int>();
            const int fig_h   = vis_cfg["fig_height"].as<int>();
            const float az    = vis_cfg["view_azimuth"].as<float>();
            const float el    = vis_cfg["view_elevation"].as<float>();
            const float msz   = vis_cfg["marker_size"].as<float>();

            auto fig = matplot::figure(true);
            fig->width(fig_w);
            fig->height(fig_h);
            auto ax = fig->current_axes();
            ax->hold(true);

            // -- Draw bounding box edges in dashed gray --
            {
                double x0 = bbox_min(0), x1 = bbox_max(0);
                double y0 = bbox_min(1), y1 = bbox_max(1);
                double z0 = bbox_min(2), z1 = bbox_max(2);
                // 12 edges of the box
                std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges = {
                    {{x0,y0,z0},{x1,y0,z0}}, {{x0,y1,z0},{x1,y1,z0}},
                    {{x0,y0,z1},{x1,y0,z1}}, {{x0,y1,z1},{x1,y1,z1}},
                    {{x0,y0,z0},{x0,y1,z0}}, {{x1,y0,z0},{x1,y1,z0}},
                    {{x0,y0,z1},{x0,y1,z1}}, {{x1,y0,z1},{x1,y1,z1}},
                    {{x0,y0,z0},{x0,y0,z1}}, {{x1,y0,z0},{x1,y0,z1}},
                    {{x0,y1,z0},{x0,y1,z1}}, {{x1,y1,z0},{x1,y1,z1}},
                };
                for (const auto& [a, b] : edges) {
                    auto l = ax->plot3({a(0), b(0)}, {a(1), b(1)}, {a(2), b(2)}, "--");
                    l->color({0.f, 0.6f, 0.6f, 0.6f});
                    l->line_width(1.0f);
                }
            }

            // -- Measurement points (red) --
            {
                const auto& m = measurements[k];
                if (m.cols() > 0) {
                    std::vector<double> mx(m.cols()), my(m.cols()), mz(m.cols());
                    for (int j = 0; j < m.cols(); ++j) {
                        mx[j] = m(0, j); my[j] = m(1, j); mz[j] = m(2, j);
                    }
                    auto mp = ax->plot3(mx, my, mz, ".");
                    mp->color({0.f, 1.0f, 0.0f, 0.0f});
                    mp->marker_size(msz);
                }
            }

            // -- Truth wireframes (black) --
            for (std::size_t t = 0; t < truth_targets.size(); ++t) {
                const auto& tgt = truth_targets[t];
                if (k < tgt.birth_time || k > tgt.death_time) continue;
                int idx = k - tgt.birth_time;
                Eigen::Vector3d pos = tgt.states[idx].head(3);
                const Eigen::Matrix3d& R = tgt.rotations[idx];

                const Eigen::MatrixXd& tris = target_mesh_tris[t];
                int n_tris = static_cast<int>(tris.cols()) / 3;

                std::vector<double> wx, wy, wz;
                for (int tr = 0; tr < n_tris; ++tr) {
                    Eigen::Vector3d v0 = R * tris.col(3*tr)   + pos;
                    Eigen::Vector3d v1 = R * tris.col(3*tr+1) + pos;
                    Eigen::Vector3d v2 = R * tris.col(3*tr+2) + pos;
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(v1(0)); wy.push_back(v1(1)); wz.push_back(v1(2));
                    wx.push_back(v2(0)); wy.push_back(v2(1)); wz.push_back(v2(2));
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(std::nan("")); wy.push_back(std::nan(""));
                    wz.push_back(std::nan(""));
                }
                auto wl = ax->plot3(wx, wy, wz);
                wl->color({0.f, 0.f, 0.f, 0.f});
                wl->line_width(0.3f);
            }

            // -- Estimated wireframes (colored) --
            for (std::size_t i = 0; i < latest.size(); ++i) {
                const auto& est = latest.component(i);
                Eigen::Vector3d est_pos = est.mean().head(3);
                Eigen::Matrix3d est_R   = Eigen::Matrix3d(est.rotation());

                // Find which template this estimate uses
                const Eigen::MatrixXd* tri_ptr = nullptr;
                int n_tris = 0;
                for (std::size_t ti = 0; ti < templates.size(); ++ti) {
                    if (&est.get_template() == templates[ti].templ.get()) {
                        tri_ptr = &templates[ti].body_triangles;
                        n_tris  = static_cast<int>(tri_ptr->cols()) / 3;
                        break;
                    }
                }
                if (!tri_ptr) continue;

                std::vector<double> wx, wy, wz;
                for (int tr = 0; tr < n_tris; ++tr) {
                    Eigen::Vector3d v0 = est_R * tri_ptr->col(3*tr)   + est_pos;
                    Eigen::Vector3d v1 = est_R * tri_ptr->col(3*tr+1) + est_pos;
                    Eigen::Vector3d v2 = est_R * tri_ptr->col(3*tr+2) + est_pos;
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(v1(0)); wy.push_back(v1(1)); wz.push_back(v1(2));
                    wx.push_back(v2(0)); wy.push_back(v2(1)); wz.push_back(v2(2));
                    wx.push_back(v0(0)); wy.push_back(v0(1)); wz.push_back(v0(2));
                    wx.push_back(std::nan("")); wy.push_back(std::nan(""));
                    wz.push_back(std::nan(""));
                }
                auto c = brew::plot_utils::lines_color(static_cast<int>(i));
                auto wl = ax->plot3(wx, wy, wz);
                wl->color(c);
                wl->line_width(0.3f);

                // Estimated position marker
                std::vector<double> epx = {est_pos(0)};
                std::vector<double> epy = {est_pos(1)};
                std::vector<double> epz = {est_pos(2)};
                auto ep = ax->plot3(epx, epy, epz, "o");
                ep->color(c);
                ep->marker_size(msz);
                ep->marker_face_color(c);
                ep->marker_face(true);
            }

            // -- Truth position markers (black X) --
            for (std::size_t t = 0; t < truth_targets.size(); ++t) {
                const auto& tgt = truth_targets[t];
                if (k < tgt.birth_time || k > tgt.death_time) continue;
                int idx = k - tgt.birth_time;
                Eigen::Vector3d pos = tgt.states[idx].head(3);
                std::vector<double> tpx = {pos(0)};
                std::vector<double> tpy = {pos(1)};
                std::vector<double> tpz = {pos(2)};
                auto tp = ax->plot3(tpx, tpy, tpz, "x");
                tp->color({0.f, 0.f, 0.f, 0.f});
                tp->marker_size(msz + 2.0f);
                tp->line_width(2.0f);
            }

            ax->xlim({bbox_min(0), bbox_max(0)});
            ax->ylim({bbox_min(1), bbox_max(1)});
            ax->zlim({bbox_min(2), bbox_max(2)});
            ax->xlabel("X"); ax->ylabel("Y"); ax->zlabel("Z");
            ax->x_axis().label_font_size(14);
            ax->y_axis().label_font_size(14);
            ax->z_axis().label_font_size(14);
            ax->title("Timestep " + std::to_string(k));
            ax->view(az, el);

            std::string fname = output_dir + "/step_"
                + std::to_string(k) + ".png";
            brew::plot_utils::save_figure(fig, fname);
            std::cout << "  -> saved " << fname << "\n";
        }
#endif
    }

    std::cout << "\nDone.\n";
    return 0;
}
