#include <gtest/gtest.h>
#include "brew/multi_target/phd.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/dynamics/integrator_2d.hpp"
#include <random>
#include <vector>
#include <limits>

using namespace brew;

struct TruthTarget {
    int birth_time;
    int death_time;
    std::vector<Eigen::VectorXd> states;
};

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

static Eigen::MatrixXd generate_measurements_with_clutter(
    const std::vector<TruthTarget>& targets,
    int timestep,
    double meas_std,
    std::mt19937& rng,
    double p_detect,
    int clutter_rate,
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

    if (meas_list.empty()) {
        return Eigen::MatrixXd(2, 0);
    }

    Eigen::MatrixXd Z(2, static_cast<int>(meas_list.size()));
    for (int j = 0; j < static_cast<int>(meas_list.size()); ++j) {
        Z.col(j) = meas_list[j];
    }
    return Z;
}

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

TEST(PHDTracking, GMWithClutterAndDropouts) {
    std::mt19937 rng(99);

    const double dt = 1.0;
    const int num_steps = 36;
    const double meas_std = 1.5;
    const double p_detect = 0.85;

    auto dyn = std::make_shared<dynamics::Integrator2D>();

    Eigen::VectorXd x0_a(4), x0_b(4);
    x0_a << -15.0, -10.0, 2.0, 1.2;
    x0_b << 55.0, 5.0, -1.6, 0.8;

    auto target_a = make_linear_target(x0_a, 0, num_steps - 1, dt, *dyn);
    auto target_b = make_linear_target(x0_b, 0, num_steps - 1, dt, *dyn);

    std::vector<TruthTarget> targets = {target_a, target_b};

    auto ekf = std::make_unique<filters::EKF>();
    ekf->set_dynamics(dyn);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    ekf->set_measurement_jacobian(H);
    ekf->set_process_noise(0.8 * Eigen::MatrixXd::Identity(4, 4));
    ekf->set_measurement_noise(meas_std * meas_std * Eigen::MatrixXd::Identity(2, 2));

    auto birth = std::make_unique<distributions::Mixture<distributions::Gaussian>>();
    Eigen::MatrixXd birth_cov = Eigen::MatrixXd::Identity(4, 4);
    birth_cov(0, 0) = 250.0; birth_cov(1, 1) = 250.0;
    birth_cov(2, 2) = 15.0; birth_cov(3, 3) = 15.0;

    Eigen::VectorXd b1(4), b2(4);
    b1 << -15.0, -10.0, 0.0, 0.0;
    b2 << 55.0, 5.0, 0.0, 0.0;
    birth->add_component(std::make_unique<distributions::Gaussian>(b1, birth_cov), 0.08);
    birth->add_component(std::make_unique<distributions::Gaussian>(b2, birth_cov), 0.08);

    auto intensity = std::make_unique<distributions::Mixture<distributions::Gaussian>>();

    multi_target::PHD<distributions::Gaussian> phd;
    phd.set_filter(std::move(ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(p_detect);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(6.0);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(40);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);

    Eigen::Vector2d clutter_min(-25.0, -25.0);
    Eigen::Vector2d clutter_max(70.0, 70.0);

    int good_steps = 0;
    int eval_steps = 0;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);

        Eigen::MatrixXd meas = generate_measurements_with_clutter(
            targets, k, meas_std, rng, p_detect, 6, clutter_min, clutter_max);

        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        if (k >= 12) {
            eval_steps++;
            Eigen::VectorXd truth_a = target_a.states[k].head(2);
            Eigen::VectorXd truth_b = target_b.states[k].head(2);
            double err_a = closest_estimate_error(latest, truth_a);
            double err_b = closest_estimate_error(latest, truth_b);
            if (latest.size() >= 2 && err_a < 7.0 && err_b < 7.0) {
                good_steps++;
            }
        }
    }

    EXPECT_GE(good_steps, 8) << "GM-PHD should maintain both tracks despite clutter";
    EXPECT_GE(eval_steps, 1);
}

TEST(PHDTracking, GGIWExtendedSparseMeasurements) {
    std::mt19937 rng(1234);

    const double dt = 1.0;
    const int num_steps = 22;
    const double meas_std = 0.7;

    auto dyn = std::make_shared<dynamics::Integrator2D>();

    Eigen::VectorXd x0(4);
    x0 << -8.0, -12.0, 2.1, 0.9;
    auto target = make_linear_target(x0, 0, num_steps - 1, dt, *dyn);

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

    auto birth = std::make_unique<distributions::Mixture<distributions::GGIW>>();
    Eigen::VectorXd b_mean(4);
    b_mean << -8.0, -12.0, 0.0, 0.0;
    Eigen::MatrixXd b_cov = 150.0 * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd b_V = 18.0 * Eigen::MatrixXd::Identity(2, 2);
    birth->add_component(
        std::make_unique<distributions::GGIW>(b_mean, b_cov, 10.0, 1.0, 12.0, b_V), 0.12);

    auto intensity = std::make_unique<distributions::Mixture<distributions::GGIW>>();

    multi_target::PHD<distributions::GGIW> phd;
    phd.set_filter(std::move(ggiw_ekf));
    phd.set_birth_model(std::move(birth));
    phd.set_intensity(std::move(intensity));
    phd.set_prob_detection(0.9);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(0.5);
    phd.set_clutter_density(1e-4);
    phd.set_prune_threshold(1e-5);
    phd.set_merge_threshold(4.0);
    phd.set_max_components(20);
    phd.set_extract_threshold(0.4);
    phd.set_gate_threshold(25.0);
    phd.set_extended_target(true);

    std::normal_distribution<double> noise(0.0, meas_std);
    Eigen::Matrix2d extent;
    extent << 4.0, 1.0, 1.0, 2.2;
    Eigen::LLT<Eigen::MatrixXd> llt(extent);
    Eigen::MatrixXd L_ext = llt.matrixL();

    int tracked_steps = 0;

    for (int k = 0; k < num_steps; ++k) {
        phd.predict(k, dt);

        int num_meas = 2 + (rng() % 4);
        Eigen::MatrixXd meas(2, num_meas);
        Eigen::VectorXd truth_pos = target.states[k].head(2);

        for (int j = 0; j < num_meas; ++j) {
            Eigen::Vector2d z(noise(rng), noise(rng));
            meas.col(j) = truth_pos + L_ext * z;
        }

        phd.correct(meas);
        phd.cleanup();

        const auto& extracted = phd.extracted_mixtures();
        const auto& latest = *extracted.back();

        double err = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < latest.size(); ++i) {
            Eigen::VectorXd est_pos = latest.component(i).mean().head(2);
            err = std::min(err, (est_pos - truth_pos).norm());
        }

        if (k >= 6 && latest.size() >= 1 && err < 9.0) {
            tracked_steps++;
        }
    }

    EXPECT_GE(tracked_steps, 6) << "GGIW-PHD should track with sparse measurements";
}
