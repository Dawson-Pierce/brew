#include <gtest/gtest.h>
#include "brew/trajectory_usggiw/trajectory_usggiw.hpp"

using Inner = brew::models::USGGIW<>;
using Trajectory = brew::models::TrajectoryUSGGIW<>;
using Filter = brew::filters::TrajectoryUSGGIWEKF<>;

static Inner make_spherical_state(const Eigen::Vector3d& direction,
                                  const Eigen::Vector2d& velocity) {
    return Inner(12.0, 2.0, direction, velocity,
                 1e-3 * Eigen::Matrix4d::Identity(),
                 12.0, 0.01 * Eigen::Matrix2d::Identity());
}

TEST(TrajectoryUSGGIW, PredictionMaintainsSphericalTrajectory) {
    Filter filter;
    filter.set_process_noise(1e-6 * Eigen::Matrix4d::Identity());
    Trajectory trajectory(4, make_spherical_state(
        Eigen::Vector3d::UnitX(), {0.01, -0.002}), 8);
    for (int i = 0; i < 12; ++i) trajectory = filter.predict(1.0, trajectory);
    EXPECT_EQ(trajectory.window_size(), 8);
    EXPECT_NEAR(trajectory.current().direction().norm(), 1.0, 1e-12);
    EXPECT_GE(trajectory.state_history_directions().cols(), 12);
}

TEST(TrajectoryUSGGIW, ComponentCorrectionUsesReportState) {
    Filter filter;
    filter.set_process_noise(Eigen::Matrix4d::Zero());
    const Eigen::Vector2d truth_rate(0.01, 0.0);
    Trajectory trajectory(4, make_spherical_state(
        Eigen::Vector3d::UnitX(), truth_rate), 4);
    trajectory = filter.predict(1.0, trajectory);
    const Eigen::Vector3d truth = brew::usggiw::propagate(
        Eigen::Vector3d::UnitX(), truth_rate, 1.0).direction;
    Inner report = make_spherical_state(truth, Eigen::Vector2d(0.02, -0.01));
    const auto corrected = filter.correct_component(report, trajectory).distribution;
    EXPECT_GT(corrected.current().angular_velocity()(0), truth_rate(0));
    EXPECT_LT(corrected.current().angular_velocity()(1), 0.0);
}

TEST(TrajectoryUSGGIW, ComponentCorrectionUpdatesRetainedTrajectory) {
    Filter filter;
    filter.set_process_noise(1e-6 * Eigen::Matrix4d::Identity());
    const Eigen::Vector2d velocity(0.01, -0.002);
    Trajectory trajectory(4, make_spherical_state(
        Eigen::Vector3d::UnitX(), velocity), 4);
    trajectory = filter.predict(1.0, trajectory);
    trajectory = filter.predict(1.0, trajectory);
    const Eigen::Vector3d retained_before = trajectory.history_at(0).direction();
    Inner report = make_spherical_state(
        brew::usggiw::apply_position_error(
            trajectory.current().direction(), {0.02, -0.01}),
        trajectory.current().angular_velocity());

    const auto corrected = filter.correct_component(report, trajectory).distribution;
    const Eigen::Vector3d retained_after = corrected.history_at(0).direction();

    EXPECT_EQ(corrected.window_directions().cols(), corrected.window_size());
    EXPECT_GT(std::acos(std::clamp(
        retained_before.dot(retained_after), -1.0, 1.0)), 1e-8);
}

TEST(TrajectoryUSGGIW, MergePreservesLongerStoredHistory) {
    Filter filter;
    filter.set_process_noise(Eigen::Matrix4d::Zero());
    const Inner initial = make_spherical_state(
        Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    Trajectory mature(4, initial, 4);
    Trajectory recent(4, initial, 4);
    for (int i = 0; i < 12; ++i) mature = filter.predict(1.0, mature);
    for (int i = 0; i < 4; ++i) recent = filter.predict(1.0, recent);
    const int mature_history = mature.state_history_directions().cols();
    ASSERT_GT(mature_history, recent.state_history_directions().cols());
    ASSERT_EQ(mature.window_size(), recent.window_size());

    brew::models::Mixture<Trajectory> mixture;
    mixture.add_component(mature, 0.2);
    mixture.add_component(recent, 0.8);
    brew::trajectory_usggiw::merge(mixture, 1.0);

    ASSERT_EQ(mixture.size(), 1u);
    EXPECT_EQ(mixture.component(0).state_history_directions().cols(), mature_history);
    EXPECT_NEAR(mixture.weight(0), 1.0, 1e-12);
}

TEST(TrajectoryUSGGIW, PhdCoastsAndReacquiresAcrossUnobservedWindow) {
    Filter filter;
    filter.set_process_noise(Eigen::DiagonalMatrix<double, 4>(
        1e-6, 1e-6, 1e-8, 1e-8).toDenseMatrix());
    filter.set_measurement_noise(1e-5 * Eigen::Matrix2d::Identity());
    filter.set_ci_weight(0.5);

    brew::trajectory_usggiw::PHD<> phd;
    phd.set_filter(filter.clone());
    phd.set_prob_detection(0.9);
    phd.set_prob_survive(1.0);
    phd.set_clutter_rate(1e-5);
    phd.set_clutter_density(1.0);
    phd.set_extract_threshold(0.2);
    phd.set_prune_threshold(1e-8);
    phd.set_merge_threshold(25.0);
    phd.set_gate_threshold(36.0);

    const Eigen::Vector2d velocity(0.01, -0.002);
    Inner initial = make_spherical_state(Eigen::Vector3d::UnitX(), velocity);
    brew::models::Mixture<Trajectory> intensity;
    intensity.add_component(Trajectory(4, initial, 8), 0.9);
    phd.set_intensity(intensity.clone());

    Eigen::MatrixXd hidden_observer(3, 1);
    hidden_observer.col(0) = -Eigen::Vector3d::UnitX();
    phd.set_detection_field(hidden_observer, 0.1, 0.0);
    Eigen::Vector3d truth = Eigen::Vector3d::UnitX();
    Eigen::Vector2d truth_velocity = velocity;
    for (int frame = 0; frame < 20; ++frame) {
        const auto propagated = brew::usggiw::propagate(truth, truth_velocity, 1.0);
        truth = propagated.direction;
        truth_velocity = propagated.velocity;
        phd.predict(frame, 1.0);
        phd.correct(Eigen::MatrixXd(3, 0));
        phd.cleanup();
        ASSERT_FALSE(phd.extracted_mixtures().empty());
        ASSERT_EQ(phd.extracted_mixtures().back()->size(), 1u);
    }

    Inner report = make_spherical_state(
        brew::usggiw::apply_position_error(truth, {0.005, -0.003}), truth_velocity);
    brew::models::Mixture<Inner> reports;
    reports.add_component(report, 1.0);
    Eigen::MatrixXd visible_observer(3, 1);
    visible_observer.col(0) = truth;
    phd.set_detection_field(visible_observer, 0.2, 0.0);
    phd.set_measurement_components(reports);
    Eigen::MatrixXd measurement(3, 1);
    measurement.col(0) = report.direction();
    phd.predict(20, 1.0);
    const auto propagated = brew::usggiw::propagate(truth, truth_velocity, 1.0);
    truth = propagated.direction;
    truth_velocity = propagated.velocity;
    phd.correct(measurement);
    phd.cleanup();

    const auto& latest = *phd.extracted_mixtures().back();
    ASSERT_GE(latest.size(), 1u);
    const double error = std::acos(std::clamp(
        latest.component(0).current().direction().dot(truth), -1.0, 1.0));
    EXPECT_LT(error, 0.03);
    EXPECT_GE(latest.component(0).state_history_directions().cols(), 20);
}

TEST(TrajectoryUSGGIW, RepeatedCoverageFromOneRingCountsOnce) {
    Filter filter;
    filter.set_process_noise(Eigen::Matrix4d::Zero());

    brew::trajectory_usggiw::PHD<> phd;
    phd.set_filter(filter.clone());
    phd.set_prob_detection(0.2);
    phd.set_prob_survive(1.0);
    phd.set_extract_threshold(0.7);
    phd.set_prune_threshold(1e-8);

    brew::models::Mixture<Trajectory> intensity;
    intensity.add_component(Trajectory(4, make_spherical_state(
        Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero()), 4), 1.0);
    phd.set_intensity(intensity.clone());

    Eigen::MatrixXd field(3, 10);
    field.colwise() = Eigen::Vector3d::UnitX();
    phd.set_grouped_detection_field(field, std::vector<int>(10, 0), 0.1, 0.0);
    phd.predict(0, 1.0);
    phd.correct(Eigen::MatrixXd(3, 0));
    phd.cleanup();

    const auto& latest = *phd.extracted_mixtures().back();
    ASSERT_EQ(latest.size(), 1u);
    EXPECT_NEAR(latest.weight(0), 0.8 + 0.2 * std::pow(2.0 / 3.0, 12.0), 1e-12);
}
