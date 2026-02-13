#pragma once

#include "brew/dynamics/integrator_2d.hpp"
#include "brew/distributions/mixture.hpp"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <type_traits>

#ifdef BREW_ENABLE_PLOTTING
#include <matplot/matplot.h>
#include <brew/plot_utils/plot_options.hpp>
#include <brew/plot_utils/color_palette.hpp>
#include <brew/plot_utils/plot_gaussian.hpp>
#include <brew/plot_utils/plot_ggiw.hpp>
#include <brew/plot_utils/plot_trajectory_gaussian.hpp>
#include <brew/plot_utils/plot_trajectory_ggiw.hpp>
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
    const distributions::Mixture<T>& estimates,
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
    const distributions::Mixture<T>& estimates)
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
/// Gaussian → covariance ellipse, GGIW → extent ellipse,
/// TrajectoryGaussian → trajectory line, TrajectoryGGIW → trajectory + extent.
template <typename T>
inline void plot_distribution_2d(
    matplot::axes_handle ax,
    const T& dist,
    const std::vector<int>& plt_inds,
    const brew::plot_utils::Color& color)
{
    if constexpr (has_get_last_state<T>::value && is_ggiw_type<T>::value) {
        brew::plot_utils::plot_trajectory_ggiw_2d(ax, dist, plt_inds, color);
    } else if constexpr (has_get_last_state<T>::value) {
        brew::plot_utils::plot_trajectory_gaussian_2d(ax, dist, plt_inds, color);
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

#endif // BREW_ENABLE_PLOTTING

} // namespace brew::test
