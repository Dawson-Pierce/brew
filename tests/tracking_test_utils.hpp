#pragma once

#include "brew/dynamics/integrator_2d.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/trajectory_ggiw.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include "brew/models/trajectory_ggiw_orientation.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/filters/ggiw_orientation_ekf.hpp"
#include "brew/filters/trajectory_gaussian_ekf.hpp"
#include "brew/filters/trajectory_ggiw_ekf.hpp"
#include "brew/filters/trajectory_ggiw_orientation_ekf.hpp"
#include "brew/multi_target/phd.hpp"
#include "brew/multi_target/cphd.hpp"
#include "brew/multi_target/mbm.hpp"
#include "brew/multi_target/pmbm.hpp"
#include "brew/multi_target/mb.hpp"
#include "brew/multi_target/lmb.hpp"
#include "brew/multi_target/glmb.hpp"
#include "brew/multi_target/jglmb.hpp"
#include "brew/clustering/dbscan.hpp"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <type_traits>
#include <memory>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/plot_utils/plot_options.hpp>
#include <brew/plot_utils/color_palette.hpp>
#include <brew/plot_utils/plot_gaussian.hpp>
#include <brew/plot_utils/plot_ggiw.hpp>
#include <brew/plot_utils/plot_trajectory_gaussian.hpp>
#include <brew/plot_utils/plot_trajectory_ggiw.hpp>
#include <brew/plot_utils/plot_ggiw_orientation.hpp>
#include <brew/plot_utils/plot_trajectory_ggiw_orientation.hpp>
#include <filesystem>
#include <map>
#endif

namespace brew::test {

// ---- Truth target ----

struct TruthTarget {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states;
};

inline TruthTarget make_linear_target(
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

// ---- Scenario data ----

struct ScenarioData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<TruthTarget> targets;
    std::vector<Eigen::MatrixXd> measurements;
    int num_steps;
    double dt;
    double meas_std;
    double p_detect;

    double clutter_rate = 0.0;
    double surveillance_area = 0.0;
    Eigen::Vector2d clutter_min = Eigen::Vector2d::Zero();
    Eigen::Vector2d clutter_max = Eigen::Vector2d::Zero();

    Eigen::Matrix2d extent = Eigen::Matrix2d::Zero();
    bool is_extended = false;

    int true_cardinality(int k) const {
        int count = 0;
        for (const auto& tgt : targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) count++;
        }
        return count;
    }
};

// ---- Internal measurement generators ----

namespace detail {

inline Eigen::MatrixXd generate_point_measurements(
    const std::vector<TruthTarget>& targets,
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

inline Eigen::MatrixXd generate_measurements_with_clutter(
    const std::vector<TruthTarget>& targets,
    int timestep, double meas_std,
    std::mt19937& rng, double p_detect,
    double clutter_rate,
    const Eigen::Vector2d& clutter_min,
    const Eigen::Vector2d& clutter_max)
{
    std::normal_distribution<double> noise(0.0, meas_std);
    std::uniform_real_distribution<double> det_roll(0.0, 1.0);
    std::uniform_real_distribution<double> unif_x(clutter_min.x(), clutter_max.x());
    std::uniform_real_distribution<double> unif_y(clutter_min.y(), clutter_max.y());
    std::poisson_distribution<int> poisson(clutter_rate);

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

    int clutter_n = poisson(rng);
    for (int i = 0; i < clutter_n; ++i) {
        Eigen::VectorXd z(2);
        z(0) = unif_x(rng);
        z(1) = unif_y(rng);
        meas_list.push_back(z);
    }

    if (meas_list.empty()) return Eigen::MatrixXd(2, 0);

    Eigen::MatrixXd Z(2, static_cast<int>(meas_list.size()));
    for (int j = 0; j < static_cast<int>(meas_list.size()); ++j) {
        Z.col(j) = meas_list[j];
    }
    return Z;
}

inline Eigen::MatrixXd generate_extended_measurements(
    const TruthTarget& target,
    int timestep, double meas_std,
    std::mt19937& rng,
    const Eigen::MatrixXd& L_ext)
{
    std::normal_distribution<double> noise(0.0, meas_std);
    int idx = timestep - target.birth_time;
    Eigen::VectorXd truth_pos = target.states[idx].head(2);

    int num_meas = 3 + static_cast<int>(rng() % 5); // 3-7 measurements
    Eigen::MatrixXd meas(2, num_meas);

    for (int j = 0; j < num_meas; ++j) {
        Eigen::VectorXd z(2);
        z(0) = noise(rng);
        z(1) = noise(rng);
        meas.col(j) = truth_pos + L_ext * z;
    }

    return meas;
}

} // namespace detail

// ---- Scenario factories ----

inline ScenarioData make_two_point_targets_scenario() {
    ScenarioData s;
    s.num_steps = 30;
    s.dt = 1.0;
    s.meas_std = 0.3;
    s.p_detect = 0.98;

    auto dyn = dynamics::Integrator2D();

    Eigen::VectorXd x0_a(4), x0_b(4);
    x0_a << 0.0, 0.0, 2.0, 1.0;
    x0_b << 50.0, 0.0, -1.0, 1.5;

    s.targets.push_back(make_linear_target(x0_a, 0, s.num_steps - 1, s.dt, dyn));
    s.targets.push_back(make_linear_target(x0_b, 0, s.num_steps - 1, s.dt, dyn));

    std::mt19937 rng(42);
    s.measurements.resize(s.num_steps);
    for (int k = 0; k < s.num_steps; ++k) {
        s.measurements[k] = detail::generate_point_measurements(
            s.targets, k, s.meas_std, rng, s.p_detect);
    }

    return s;
}

inline ScenarioData make_extended_target_scenario() {
    ScenarioData s;
    s.num_steps = 20;
    s.dt = 1.0;
    s.meas_std = 0.3;
    s.p_detect = 0.95;
    s.is_extended = true;
    s.extent << 4.0, 1.0, 1.0, 2.0;

    auto dyn = dynamics::Integrator2D();

    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.0, 2.0, 0.5;
    s.targets.push_back(make_linear_target(x0, 0, s.num_steps - 1, s.dt, dyn));

    Eigen::LLT<Eigen::Matrix2d> llt(s.extent);
    Eigen::Matrix2d L_ext = llt.matrixL();

    std::mt19937 rng(123);
    s.measurements.resize(s.num_steps);
    for (int k = 0; k < s.num_steps; ++k) {
        s.measurements[k] = detail::generate_extended_measurements(
            s.targets[0], k, s.meas_std, rng, L_ext);
    }

    return s;
}

inline ScenarioData make_clutter_scenario() {
    ScenarioData s;
    s.num_steps = 36;
    s.dt = 1.0;
    s.meas_std = 0.5;
    s.p_detect = 0.90;
    s.clutter_rate = 4.0;
    s.clutter_min = Eigen::Vector2d(-25.0, -25.0);
    s.clutter_max = Eigen::Vector2d(70.0, 70.0);
    s.surveillance_area = (s.clutter_max.x() - s.clutter_min.x()) *
                          (s.clutter_max.y() - s.clutter_min.y());

    auto dyn = dynamics::Integrator2D();

    Eigen::VectorXd x0_a(4), x0_b(4);
    x0_a << -15.0, -10.0, 2.0, 1.2;
    x0_b << 55.0, 5.0, -1.6, 0.8;

    s.targets.push_back(make_linear_target(x0_a, 0, s.num_steps - 1, s.dt, dyn));
    s.targets.push_back(make_linear_target(x0_b, 0, s.num_steps - 1, s.dt, dyn));

    std::mt19937 rng(99);
    s.measurements.resize(s.num_steps);
    for (int k = 0; k < s.num_steps; ++k) {
        s.measurements[k] = detail::generate_measurements_with_clutter(
            s.targets, k, s.meas_std, rng, s.p_detect,
            s.clutter_rate, s.clutter_min, s.clutter_max);
    }

    return s;
}

inline ScenarioData make_variable_targets_scenario() {
    ScenarioData s;
    s.num_steps = 30;
    s.dt = 1.0;
    s.meas_std = 0.3;
    s.p_detect = 0.98;

    auto dyn = dynamics::Integrator2D();

    // Target A: born at k=0, dies at k=24
    Eigen::VectorXd x0_a(4);
    x0_a << 0.0, 0.0, 2.0, 1.0;
    s.targets.push_back(make_linear_target(x0_a, 0, 24, s.dt, dyn));

    // Target B: born at k=5, dies at k=28
    Eigen::VectorXd x0_b(4);
    x0_b << 50.0, 0.0, -1.0, 1.5;
    s.targets.push_back(make_linear_target(x0_b, 5, 28, s.dt, dyn));

    // Target C: born at k=10, dies at k=19
    Eigen::VectorXd x0_c(4);
    x0_c << 25.0, 30.0, 0.5, -1.0;
    s.targets.push_back(make_linear_target(x0_c, 10, 19, s.dt, dyn));

    std::mt19937 rng(77);
    s.measurements.resize(s.num_steps);
    for (int k = 0; k < s.num_steps; ++k) {
        s.measurements[k] = detail::generate_point_measurements(
            s.targets, k, s.meas_std, rng, s.p_detect);
    }

    return s;
}

// ---- Unified scenario factory (two-step: base + measurements) ----

inline ScenarioData make_base_scenario() {
    ScenarioData s;
    s.num_steps = 30;
    s.dt = 1.0;
    s.meas_std = 0.3;
    s.p_detect = 0.98;

    auto dyn = dynamics::Integrator2D();

    Eigen::VectorXd x0_a(4), x0_b(4);
    x0_a << 0.0, 0.0, 2.0, 0.5;
    x0_b << 50.0, 0.0, -0.5, 1.5;

    s.targets.push_back(make_linear_target(x0_a, 0, s.num_steps - 1, s.dt, dyn));
    s.targets.push_back(make_linear_target(x0_b, 0, s.num_steps - 1, s.dt, dyn));

    s.measurements.resize(s.num_steps);
    return s;
}

inline void generate_scenario_point_measurements(ScenarioData& s, unsigned seed = 42) {
    std::mt19937 rng(seed);
    for (int k = 0; k < s.num_steps; ++k) {
        s.measurements[k] = detail::generate_point_measurements(
            s.targets, k, s.meas_std, rng, s.p_detect);
    }
}

inline void generate_scenario_extended_measurements(ScenarioData& s, unsigned seed = 42) {
    s.is_extended = true;
    s.extent << 4.0, 1.0, 1.0, 2.0;

    Eigen::LLT<Eigen::Matrix2d> llt(s.extent);
    Eigen::Matrix2d L_ext = llt.matrixL();

    std::mt19937 rng(seed);
    for (int k = 0; k < s.num_steps; ++k) {
        std::vector<Eigen::MatrixXd> per_target_meas;
        int total_cols = 0;
        for (const auto& tgt : s.targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                auto m = detail::generate_extended_measurements(
                    tgt, k, s.meas_std, rng, L_ext);
                per_target_meas.push_back(m);
                total_cols += static_cast<int>(m.cols());
            }
        }
        if (total_cols == 0) {
            s.measurements[k] = Eigen::MatrixXd(2, 0);
        } else {
            Eigen::MatrixXd Z(2, total_cols);
            int col = 0;
            for (const auto& m : per_target_meas) {
                Z.block(0, col, 2, m.cols()) = m;
                col += static_cast<int>(m.cols());
            }
            s.measurements[k] = Z;
        }
    }
}

// ---- Position extraction (dispatches by type) ----

template <typename T, typename = void>
struct has_get_last_state : std::false_type {};

template <typename T>
struct has_get_last_state<T,
    std::void_t<decltype(std::declval<const T&>().get_last_state())>
> : std::true_type {};

template <typename T, typename = void>
struct is_ggiw_type : std::false_type {};

template <typename T>
struct is_ggiw_type<T,
    std::void_t<decltype(std::declval<const T&>().V())>
> : std::true_type {};

template <typename T>
constexpr bool is_extended_distribution_v = is_ggiw_type<T>::value;

template <typename T, typename = void>
struct is_orientation_type : std::false_type {};

template <typename T>
struct is_orientation_type<T,
    std::void_t<decltype(std::declval<const T&>().basis())>
> : std::true_type {};

template <typename T, typename = void>
struct has_track_histories : std::false_type {};

template <typename T>
struct has_track_histories<T,
    std::void_t<decltype(std::declval<const T&>().track_histories())>
> : std::true_type {};

template <typename T>
inline Eigen::VectorXd extract_position(const T& component) {
    if constexpr (has_get_last_state<T>::value) {
        return component.get_last_state().head(2);
    } else {
        return component.mean().head(2);
    }
}

// ---- Error computation ----

template <typename T>
inline double closest_estimate_error(
    const models::Mixture<T>& estimates,
    const Eigen::VectorXd& truth_pos)
{
    double min_err = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        Eigen::VectorXd est_pos = extract_position(estimates.component(i));
        double err = (est_pos - truth_pos).norm();
        min_err = std::min(min_err, err);
    }
    return min_err;
}

// ---- Logging ----

inline void print_tracking_header(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

inline void print_tracking_step(int step, int n_meas, int n_comp, int n_est,
    double err_a, double err_b)
{
    std::cout << std::setw(5) << step
              << std::setw(10) << n_meas
              << std::setw(10) << n_comp
              << std::setw(10) << n_est
              << std::setw(12) << std::fixed << std::setprecision(2) << err_a
              << std::setw(12) << std::fixed << std::setprecision(2) << err_b
              << "\n";
}

inline void print_tracking_step(int step, int n_meas, int n_comp, int n_est,
    double err)
{
    std::cout << std::setw(5) << step
              << std::setw(10) << n_meas
              << std::setw(10) << n_comp
              << std::setw(10) << n_est
              << std::setw(12) << std::fixed << std::setprecision(2)
              << (std::isfinite(err) ? err : -1.0)
              << "\n";
}

// ---- Scenario params ----

struct ScenarioParams {
    double p_detect;
    double p_survive = 0.99;
    double clutter_rate = 1.0;
    double clutter_density = 1e-4;
    double dbscan_epsilon = 5.0;
    int dbscan_min_pts = 2;
};

inline ScenarioParams make_default_params(const ScenarioData& s) {
    ScenarioParams p;
    p.p_detect = s.p_detect;
    return p;
}

// ---- EKF factory functions ----

inline std::unique_ptr<filters::EKF> make_ekf(const ScenarioData& scenario) {
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::EKF>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    // EKF uses G * Q * G^T where G is 4x2 for Integrator2D, so Q must be 2x2
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_measurement_noise(
        scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    return ekf;
}

inline std::unique_ptr<filters::GGIWEKF> make_ggiw_ekf(const ScenarioData& scenario) {
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::GGIWEKF>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(
        scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_temporal_decay(1.0);
    ekf->set_forgetting_factor(5.0);
    ekf->set_scaling_parameter(1.0);

    return ekf;
}

inline std::unique_ptr<filters::GGIWOrientationEKF> make_ggiw_orientation_ekf(const ScenarioData& scenario) {
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::GGIWOrientationEKF>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(
        scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));
    ekf->set_temporal_decay(1.0);
    ekf->set_forgetting_factor(5.0);
    ekf->set_scaling_parameter(1.0);

    return ekf;
}

inline std::unique_ptr<filters::TrajectoryGaussianEKF> make_trajectory_gaussian_ekf(
    const ScenarioData& scenario, int window = 10)
{
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::TrajectoryGaussianEKF>();
    ekf->set_dynamics(dyn);
    ekf->set_window_size(window);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(
        scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    return ekf;
}

inline std::unique_ptr<filters::TrajectoryGGIWEKF> make_trajectory_ggiw_ekf(
    const ScenarioData& scenario, int window = 10)
{
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::TrajectoryGGIWEKF>();
    ekf->set_dynamics(dyn);
    ekf->set_window_size(window);
    ekf->set_temporal_decay(1.0);
    ekf->set_forgetting_factor(5.0);
    ekf->set_scaling_parameter(1.0);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(
        scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    return ekf;
}

// ---- Birth model factory functions ----

inline std::unique_ptr<models::Mixture<models::Gaussian>>
make_gm_birth(double weight = 0.1)
{
    auto birth = std::make_unique<models::Mixture<models::Gaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 100.0; birth_cov(1, 1) = 100.0;
    birth_cov(2, 2) = 10.0; birth_cov(3, 3) = 10.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::Gaussian>(b1, birth_cov), weight);
    birth->add_component(
        std::make_unique<models::Gaussian>(b2, birth_cov), weight);

    return birth;
}

inline std::unique_ptr<models::Mixture<models::GGIW>>
make_ggiw_birth(double weight = 0.1)
{
    auto birth = std::make_unique<models::Mixture<models::GGIW>>();
    Eigen::MatrixXd b_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 20.0 * Eigen::MatrixXd::Identity(2, 2);

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::GGIW>(b1, b_cov, 10.0, 1.0, 10.0, b_V), weight);
    birth->add_component(
        std::make_unique<models::GGIW>(b2, b_cov, 10.0, 1.0, 10.0, b_V), weight);

    return birth;
}

inline std::unique_ptr<models::Mixture<models::GGIWOrientation>>
make_ggiw_orientation_birth(double weight = 0.1)
{
    auto birth = std::make_unique<models::Mixture<models::GGIWOrientation>>();
    Eigen::MatrixXd b_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 20.0 * Eigen::MatrixXd::Identity(2, 2);

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::GGIWOrientation>(b1, b_cov, 10.0, 1.0, 10.0, b_V), weight);
    birth->add_component(
        std::make_unique<models::GGIWOrientation>(b2, b_cov, 10.0, 1.0, 10.0, b_V), weight);

    return birth;
}

inline std::unique_ptr<models::Mixture<models::TrajectoryGaussian>>
make_trajectory_gaussian_birth(double weight = 0.1)
{
    auto birth = std::make_unique<models::Mixture<models::TrajectoryGaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 100.0; birth_cov(1, 1) = 100.0;
    birth_cov(2, 2) = 10.0; birth_cov(3, 3) = 10.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TrajectoryGaussian>(0, 4, b1, birth_cov), weight);
    birth->add_component(
        std::make_unique<models::TrajectoryGaussian>(0, 4, b2, birth_cov), weight);

    return birth;
}

inline std::unique_ptr<models::Mixture<models::TrajectoryGGIW>>
make_trajectory_ggiw_birth(double weight = 0.1)
{
    auto birth = std::make_unique<models::Mixture<models::TrajectoryGGIW>>();
    Eigen::MatrixXd b_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 20.0 * Eigen::MatrixXd::Identity(2, 2);

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TrajectoryGGIW>(
            0, 4, b1, b_cov, 10.0, 1.0, 10.0, b_V), weight);
    birth->add_component(
        std::make_unique<models::TrajectoryGGIW>(
            0, 4, b2, b_cov, 10.0, 1.0, 10.0, b_V), weight);

    return birth;
}

inline std::unique_ptr<filters::TrajectoryGGIWOrientationEKF> make_trajectory_ggiw_orientation_ekf(
    const ScenarioData& scenario, int window = 10)
{
    auto dyn = std::make_shared<dynamics::Integrator2D>();
    auto ekf = std::make_unique<filters::TrajectoryGGIWOrientationEKF>();
    ekf->set_dynamics(dyn);
    ekf->set_window_size(window);
    ekf->set_temporal_decay(1.0);
    ekf->set_forgetting_factor(5.0);
    ekf->set_scaling_parameter(1.0);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.5 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(
        scenario.meas_std * scenario.meas_std * Eigen::MatrixXd::Identity(2, 2));

    return ekf;
}

inline std::unique_ptr<models::Mixture<models::TrajectoryGGIWOrientation>>
make_trajectory_ggiw_orientation_birth(double weight = 0.1)
{
    auto birth = std::make_unique<models::Mixture<models::TrajectoryGGIWOrientation>>();
    Eigen::MatrixXd b_cov = 100.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 20.0 * Eigen::MatrixXd::Identity(2, 2);

    Eigen::VectorXd b1(4), b2(4);
    b1 << 0.0, 0.0, 0.0, 0.0;
    b2 << 50.0, 0.0, 0.0, 0.0;
    birth->add_component(
        std::make_unique<models::TrajectoryGGIWOrientation>(
            0, 4, b1, b_cov, 10.0, 1.0, 10.0, b_V), weight);
    birth->add_component(
        std::make_unique<models::TrajectoryGGIWOrientation>(
            0, 4, b2, b_cov, 10.0, 1.0, 10.0, b_V), weight);

    return birth;
}

// ---- RFS estimator factory functions ----

template <typename T>
inline multi_target::PHD<T> make_phd(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::PHD<T> phd;
    phd.set_filter(std::move(filter));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::make_unique<models::Mixture<T>>());
    phd.set_prob_detection(params.p_detect);
    phd.set_prob_survive(params.p_survive);
    phd.set_clutter_rate(params.clutter_rate);
    phd.set_clutter_density(params.clutter_density);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(30);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);

    if constexpr (is_extended_distribution_v<T>) {
        phd.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return phd;
}

template <typename T>
inline multi_target::CPHD<T> make_cphd(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::CPHD<T> cphd;
    cphd.set_filter(std::move(filter));
    cphd.set_birth_model(std::move(birth));
    cphd.set_intensity(std::make_unique<models::Mixture<T>>());
    cphd.set_prob_detection(params.p_detect);
    cphd.set_prob_survive(params.p_survive);
    cphd.set_clutter_rate(params.clutter_rate);
    cphd.set_clutter_density(params.clutter_density);
    cphd.set_prune_threshold(1e-5);
    cphd.set_merge_threshold(4.0);
    cphd.set_max_components(30);
    cphd.set_extract_threshold(0.4);
    cphd.set_gate_threshold(25.0);
    cphd.set_poisson_cardinality(0.1);
    cphd.set_poisson_birth_cardinality(0.1);

    if constexpr (is_extended_distribution_v<T>) {
        cphd.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return cphd;
}

template <typename T>
inline multi_target::MBM<T> make_mbm(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::MBM<T> mbm;
    mbm.set_filter(std::move(filter));
    mbm.set_birth_model(std::move(birth));
    mbm.set_prob_detection(params.p_detect);
    mbm.set_prob_survive(params.p_survive);
    mbm.set_clutter_rate(params.clutter_rate);
    mbm.set_clutter_density(params.clutter_density);
    mbm.set_prune_threshold_hypothesis(1e-4);
    mbm.set_prune_threshold_bernoulli(1e-3);
    mbm.set_max_hypotheses(50);
    mbm.set_extract_threshold(0.4);
    mbm.set_gate_threshold(25.0);
    mbm.set_k_best(5);

    if constexpr (is_extended_distribution_v<T>) {
        mbm.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return mbm;
}

template <typename T>
inline multi_target::PMBM<T> make_pmbm(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::PMBM<T> pmbm;
    pmbm.set_filter(std::move(filter));
    pmbm.set_birth_model(std::move(birth));
    pmbm.set_poisson_intensity(std::make_unique<models::Mixture<T>>());
    pmbm.set_prob_detection(params.p_detect);
    pmbm.set_prob_survive(params.p_survive);
    pmbm.set_clutter_rate(params.clutter_rate);
    pmbm.set_clutter_density(params.clutter_density);
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

    if constexpr (is_extended_distribution_v<T>) {
        pmbm.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return pmbm;
}

template <typename T>
inline multi_target::MB<T> make_mb(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::MB<T> mb;
    mb.set_filter(std::move(filter));
    mb.set_birth_model(std::move(birth));
    mb.set_prob_detection(params.p_detect);
    mb.set_prob_survive(params.p_survive);
    mb.set_clutter_rate(params.clutter_rate);
    mb.set_clutter_density(params.clutter_density);
    mb.set_prune_threshold_bernoulli(1e-3);
    mb.set_extract_threshold(0.4);
    mb.set_gate_threshold(25.0);

    if constexpr (is_extended_distribution_v<T>) {
        mb.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return mb;
}

template <typename T>
inline multi_target::LMB<T> make_lmb(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::LMB<T> lmb;
    lmb.set_filter(std::move(filter));
    lmb.set_birth_model(std::move(birth));
    lmb.set_prob_detection(params.p_detect);
    lmb.set_prob_survive(params.p_survive);
    lmb.set_clutter_rate(params.clutter_rate);
    lmb.set_clutter_density(params.clutter_density);
    lmb.set_prune_threshold_bernoulli(1e-3);
    lmb.set_extract_threshold(0.4);
    lmb.set_gate_threshold(25.0);
    lmb.set_k_best(5);

    if constexpr (is_extended_distribution_v<T>) {
        lmb.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return lmb;
}

template <typename T>
inline multi_target::GLMB<T> make_glmb(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::GLMB<T> glmb;
    glmb.set_filter(std::move(filter));
    glmb.set_birth_model(std::move(birth));
    glmb.set_prob_detection(params.p_detect);
    glmb.set_prob_survive(params.p_survive);
    glmb.set_clutter_rate(params.clutter_rate);
    glmb.set_clutter_density(params.clutter_density);
    glmb.set_prune_threshold_hypothesis(1e-4);
    glmb.set_prune_threshold_bernoulli(1e-3);
    glmb.set_max_hypotheses(50);
    glmb.set_extract_threshold(0.4);
    glmb.set_gate_threshold(25.0);
    glmb.set_k_best(5);

    if constexpr (is_extended_distribution_v<T>) {
        glmb.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return glmb;
}

template <typename T>
inline multi_target::JGLMB<T> make_jglmb(
    std::unique_ptr<filters::Filter<T>> filter,
    std::unique_ptr<models::Mixture<T>> birth,
    const ScenarioParams& params)
{
    multi_target::JGLMB<T> jglmb;
    jglmb.set_filter(std::move(filter));
    jglmb.set_birth_model(std::move(birth));
    jglmb.set_prob_detection(params.p_detect);
    jglmb.set_prob_survive(params.p_survive);
    jglmb.set_clutter_rate(params.clutter_rate);
    jglmb.set_clutter_density(params.clutter_density);
    jglmb.set_prune_threshold_hypothesis(1e-4);
    jglmb.set_prune_threshold_bernoulli(1e-3);
    jglmb.set_max_hypotheses(50);
    jglmb.set_extract_threshold(0.4);
    jglmb.set_gate_threshold(25.0);
    jglmb.set_k_best(5);

    if constexpr (is_extended_distribution_v<T>) {
        jglmb.set_cluster_object(std::make_shared<clustering::DBSCAN>(
            params.dbscan_epsilon, params.dbscan_min_pts));
    }

    return jglmb;
}

// ---- Plotting ----

#ifdef BREW_ENABLE_PLOTTING

struct TrackingPlotData {
    std::vector<std::vector<double>> truth_x, truth_y;
    std::vector<double> meas_all_x, meas_all_y;
    std::vector<double> est_all_x, est_all_y;

    explicit TrackingPlotData(int num_targets)
        : truth_x(num_targets), truth_y(num_targets) {}
};

template <typename T>
inline void accumulate_plot_step(
    TrackingPlotData& pd,
    const ScenarioData& scenario,
    int k,
    const Eigen::MatrixXd& meas,
    const models::Mixture<T>& estimates)
{
    for (std::size_t t = 0; t < scenario.targets.size(); ++t) {
        const auto& tgt = scenario.targets[t];
        if (k >= tgt.birth_time && k <= tgt.death_time) {
            int idx = k - tgt.birth_time;
            pd.truth_x[t].push_back(tgt.states[idx](0));
            pd.truth_y[t].push_back(tgt.states[idx](1));
        }
    }

    for (int j = 0; j < meas.cols(); ++j) {
        pd.meas_all_x.push_back(meas(0, j));
        pd.meas_all_y.push_back(meas(1, j));
    }

    for (std::size_t i = 0; i < estimates.size(); ++i) {
        Eigen::VectorXd pos = extract_position(estimates.component(i));
        pd.est_all_x.push_back(pos(0));
        pd.est_all_y.push_back(pos(1));
    }
}

// Plot common elements (measurements, truth, per-step estimates) onto given axes.
inline void plot_common_elements(
    matplot::axes_handle ax,
    const TrackingPlotData& pd)
{
    // Measurements (light gray dots)
    if (!pd.meas_all_x.empty()) {
        auto mp = ax->plot(pd.meas_all_x, pd.meas_all_y, ".");
        mp->color({0.f, 0.7f, 0.7f, 0.7f});
        mp->marker_size(4.0f);
    }

    // Truth trajectories (black dotted, thin)
    for (std::size_t t = 0; t < pd.truth_x.size(); ++t) {
        if (!pd.truth_x[t].empty()) {
            auto tl = ax->plot(pd.truth_x[t], pd.truth_y[t], ":");
            tl->color({0.f, 0.f, 0.f, 0.f});
            tl->line_width(1.0f);
        }
    }
}

/// Plot track trajectory lines from ancestry bookkeeping (MBM/PMBM track_histories).
/// Each track is drawn as a colored line connecting its historical states.
inline void plot_track_histories(
    matplot::axes_handle ax,
    const std::map<int, std::vector<Eigen::VectorXd>>& histories,
    int pos_x = 0, int pos_y = 1)
{
    int color_idx = 0;
    for (const auto& [id, states] : histories) {
        if (states.size() < 2) continue;
        std::vector<double> tx, ty;
        tx.reserve(states.size());
        ty.reserve(states.size());
        for (const auto& s : states) {
            tx.push_back(s(pos_x));
            ty.push_back(s(pos_y));
        }
        auto c = brew::plot_utils::lines_color(color_idx++);
        auto tl = ax->plot(tx, ty);
        tl->color({c[0], c[1], c[2], c[3]});
        tl->line_width(2.0f);
    }
}

/// Generic 2D distribution plot: auto-dispatches based on distribution type.
/// Gaussian -> covariance ellipse, GGIW -> extent ellipse,
/// GGIWOrientation -> extent ellipse + principal axes,
/// TrajectoryGaussian -> trajectory line, TrajectoryGGIW -> trajectory + extent,
/// TrajectoryGGIWOrientation -> trajectory + extent + principal axes.
template <typename T>
inline void plot_distribution_2d(
    matplot::axes_handle ax,
    const T& dist,
    const std::vector<int>& plt_inds,
    const brew::plot_utils::Color& color)
{
    if constexpr (has_get_last_state<T>::value && is_orientation_type<T>::value) {
        brew::plot_utils::plot_trajectory_ggiw_orientation_2d(ax, dist, plt_inds, color);
    } else if constexpr (has_get_last_state<T>::value && is_ggiw_type<T>::value) {
        brew::plot_utils::plot_trajectory_ggiw_2d(ax, dist, plt_inds, color);
    } else if constexpr (has_get_last_state<T>::value) {
        brew::plot_utils::plot_trajectory_gaussian_2d(ax, dist, plt_inds, color);
    } else if constexpr (is_orientation_type<T>::value) {
        brew::plot_utils::plot_ggiw_orientation_2d(ax, dist, plt_inds, color);
    } else if constexpr (is_ggiw_type<T>::value) {
        brew::plot_utils::plot_ggiw_2d(ax, dist, plt_inds, color);
    } else {
        brew::plot_utils::plot_gaussian_2d(ax, dist, plt_inds, color, 2.0, 0.3f);
    }
}

/// Plot all extracted mixtures (every timestep) with per-component coloring.
/// Use for PHD/CPHD where per-timestep evolution is the primary visual.
template <typename MixtureContainer>
inline void plot_all_extracted(
    matplot::axes_handle ax,
    const MixtureContainer& extracted,
    const std::vector<int>& plt_inds = {0, 1})
{
    for (const auto& mix_ptr : extracted) {
        for (std::size_t i = 0; i < mix_ptr->size(); ++i) {
            auto c = brew::plot_utils::lines_color(static_cast<int>(i));
            plot_distribution_2d(ax, mix_ptr->component(i), plt_inds, c);
        }
    }
}

/// Plot only the final extracted mixture with per-component coloring.
/// Use for MBM/PMBM (track histories show temporal evolution) and
/// trajectory distributions (the distribution itself encodes history).
template <typename MixtureContainer>
inline void plot_final_extracted(
    matplot::axes_handle ax,
    const MixtureContainer& extracted,
    const std::vector<int>& plt_inds = {0, 1})
{
    if (extracted.empty()) return;
    const auto& mix_ptr = extracted.back();
    for (std::size_t i = 0; i < mix_ptr->size(); ++i) {
        auto c = brew::plot_utils::lines_color(static_cast<int>(i));
        plot_distribution_2d(ax, mix_ptr->component(i), plt_inds, c);
    }
}

// ---- Plot helper functions for test files ----

inline std::string output_dir() { return "output"; }

/// Populate a single subplot axes with the appropriate plot elements for an estimator.
/// MBM/PMBM (has track_histories): track history lines + final extracted distributions.
/// Trajectory PHD/CPHD (is_trajectory=true): final extracted (trajectory encodes history).
/// Plain PHD/CPHD: all extracted distributions across all timesteps.
template <typename Estimator>
inline void populate_estimator_axes(
    matplot::axes_handle ax,
    Estimator& est,
    const TrackingPlotData& plot_data,
    const std::string& subtitle,
    bool is_trajectory = false)
{
    plot_common_elements(ax, plot_data);

    if constexpr (has_track_histories<Estimator>::value) {
        plot_track_histories(ax, est.track_histories());
        plot_final_extracted(ax, est.extracted_mixtures());
    } else {
        if (is_trajectory) {
            plot_final_extracted(ax, est.extracted_mixtures());
        } else {
            plot_all_extracted(ax, est.extracted_mixtures());
        }
    }

    ax->title(subtitle);
    ax->xlabel("x");
    ax->ylabel("y");
}

/// Create a figure for comparison layout.
inline matplot::figure_handle create_comparison_figure(int width = 2400, int height = 1200) {
    std::filesystem::create_directories(output_dir());
    auto fig = matplot::figure(true);
    fig->width(width);
    fig->height(height);
    return fig;
}

/// Create a subplot axes at the given index in a rows x cols grid.
inline matplot::axes_handle comparison_subplot(
    matplot::figure_handle fig, int rows, int cols, int index)
{
    auto ax = matplot::subplot(fig, rows, cols, index);
    ax->hold(true);
    return ax;
}

inline void plot_cardinality_comparison(
    const ScenarioData& scenario,
    const std::vector<double>& est_card_vec,
    const std::string& title, const std::string& filename)
{
    std::filesystem::create_directories(output_dir());
    auto fig = matplot::figure(true);
    fig->width(800); fig->height(400);
    auto ax = fig->current_axes();
    ax->hold(true);

    std::vector<double> steps_vec, true_card_vec;
    for (int k = 0; k < scenario.num_steps; ++k) {
        steps_vec.push_back(static_cast<double>(k));
        true_card_vec.push_back(static_cast<double>(scenario.true_cardinality(k)));
    }

    auto tc = ax->plot(steps_vec, true_card_vec, "--");
    tc->color({0.f, 0.f, 0.f, 0.f});
    tc->line_width(2.0f);

    auto ec = ax->plot(steps_vec, est_card_vec);
    ec->color({0.f, 0.f, 0.4470f, 0.7410f});
    ec->line_width(2.0f);

    ax->title(title);
    ax->xlabel("Time step");
    ax->ylabel("Cardinality");
    brew::plot_utils::save_figure(fig, output_dir() + "/" + filename);
}

#endif // BREW_ENABLE_PLOTTING

// ---- Tracking result and generic tracking loop ----

struct TrackingResult {
    int converged_steps = 0;
#ifdef BREW_ENABLE_PLOTTING
    TrackingPlotData plot_data;
    explicit TrackingResult(int num_targets) : plot_data(num_targets) {}
#else
    explicit TrackingResult(int /*num_targets*/) {}
#endif
};

template <typename Estimator, typename T>
inline TrackingResult run_tracking(
    Estimator& est, const ScenarioData& scenario,
    const std::string& label,
    double error_threshold = 5.0, int warmup_steps = 10)
{
    TrackingResult result(static_cast<int>(scenario.targets.size()));

    print_tracking_header(label);

    for (int k = 0; k < scenario.num_steps; ++k) {
        est.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        est.correct(meas);
        est.cleanup();

        const auto& extracted = est.extracted_mixtures();
        const auto& latest = *extracted.back();

        bool all_good = (k >= warmup_steps);
        for (const auto& tgt : scenario.targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                double err = closest_estimate_error(latest, truth_pos);
                if (err >= error_threshold) all_good = false;
            }
        }

        if (all_good) result.converged_steps++;

#ifdef BREW_ENABLE_PLOTTING
        accumulate_plot_step(result.plot_data, scenario, k, meas, latest);
#endif
    }

    std::cout << "Converged steps (k>=" << warmup_steps
              << ", err<" << error_threshold << "): "
              << result.converged_steps << " / "
              << (scenario.num_steps - warmup_steps) << "\n";

    return result;
}

// ---- CPHD tracking loop (also collects estimated cardinality) ----

struct CPHDTrackingResult {
    int converged_steps = 0;
    std::vector<double> cardinality;
#ifdef BREW_ENABLE_PLOTTING
    TrackingPlotData plot_data;
    explicit CPHDTrackingResult(int num_targets) : plot_data(num_targets) {}
#else
    explicit CPHDTrackingResult(int /*num_targets*/) {}
#endif
};

template <typename Estimator, typename T>
inline CPHDTrackingResult run_tracking_cphd(
    Estimator& est, const ScenarioData& scenario,
    const std::string& label,
    double error_threshold = 5.0, int warmup_steps = 10)
{
    CPHDTrackingResult result(static_cast<int>(scenario.targets.size()));

    print_tracking_header(label);

    for (int k = 0; k < scenario.num_steps; ++k) {
        est.predict(k, scenario.dt);

        const auto& meas = scenario.measurements[k];
        est.correct(meas);
        est.cleanup();

        const auto& extracted = est.extracted_mixtures();
        const auto& latest = *extracted.back();

        bool all_good = (k >= warmup_steps);
        for (const auto& tgt : scenario.targets) {
            if (k >= tgt.birth_time && k <= tgt.death_time) {
                int idx = k - tgt.birth_time;
                Eigen::VectorXd truth_pos = tgt.states[idx].head(2);
                double err = closest_estimate_error(latest, truth_pos);
                if (err >= error_threshold) all_good = false;
            }
        }

        if (all_good) result.converged_steps++;
        result.cardinality.push_back(est.estimated_cardinality());

#ifdef BREW_ENABLE_PLOTTING
        accumulate_plot_step(result.plot_data, scenario, k, meas, latest);
#endif
    }

    std::cout << "Converged steps (k>=" << warmup_steps
              << ", err<" << error_threshold << "): "
              << result.converged_steps << " / "
              << (scenario.num_steps - warmup_steps) << "\n";

    return result;
}

} // namespace brew::test
