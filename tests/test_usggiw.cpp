#include <gtest/gtest.h>
#include "brew/usggiw/usggiw.hpp"

using Model = brew::models::USGGIW<>;
using Filter = brew::filters::USGGIWEKF<>;

static Model make_model(const Eigen::Vector3d& r, const Eigen::Vector2d& w) {
    return Model(10.0, 2.0, r, w, 0.01 * Eigen::Matrix4d::Identity(),
                 12.0, Eigen::Matrix2d::Identity());
}

static Eigen::Vector3d lonlat(double longitude, double latitude) {
    return {std::cos(latitude) * std::cos(longitude),
            std::cos(latitude) * std::sin(longitude),
            std::sin(latitude)};
}

static Eigen::Matrix4d finite_difference_transition_jacobian(
    const Eigen::Vector3d& direction, const Eigen::Vector2d& velocity, double dt) {
    const auto nominal = brew::usggiw::propagate(direction, velocity, dt);
    Eigen::Matrix4d F;
    constexpr double step = 1e-4;
    for (int column = 0; column < 4; ++column) {
        Eigen::Vector4d perturbation = Eigen::Vector4d::Zero();
        perturbation(column) = step;
        auto evaluate = [&](double sign) {
            const Eigen::Vector2d position_delta = sign * perturbation.head<2>();
            const Eigen::Vector2d velocity_delta = sign * perturbation.tail<2>();
            const Eigen::Vector3d perturbed_direction =
                brew::usggiw::apply_position_error(direction, position_delta);
            const Eigen::Vector2d perturbed_velocity =
                brew::usggiw::transport_matrix(direction, perturbed_direction)
                * (velocity + velocity_delta);
            const auto result = brew::usggiw::propagate(
                perturbed_direction, perturbed_velocity, dt);
            Eigen::Vector4d error;
            error.head<2>() = brew::usggiw::local_error(
                nominal.direction, result.direction);
            error.tail<2>() = brew::usggiw::transport_matrix(
                result.direction, nominal.direction) * result.velocity
                - nominal.velocity;
            return error;
        };
        F.col(column) = (evaluate(1.0) - evaluate(-1.0)) / (2.0 * step);
    }
    return F;
}

TEST(USGGIW, ExponentialAndLogarithmAreInverse) {
    const Eigen::Vector3d direction = Eigen::Vector3d(0.3, -0.4, 0.8).normalized();
    const Eigen::Vector2d error(0.21, -0.13);
    const Eigen::Vector3d displaced = brew::usggiw::apply_position_error(direction, error);
    EXPECT_TRUE(brew::usggiw::local_error(direction, displaced).isApprox(error, 1e-12));
}

TEST(USGGIW, ParallelTransportPreservesTangentMetric) {
    const Eigen::Vector3d from = Eigen::Vector3d(0.7, 0.2, 0.4).normalized();
    const Eigen::Vector3d to = brew::usggiw::apply_position_error(from, {0.4, -0.2});
    const Eigen::Matrix2d T = brew::usggiw::transport_matrix(from, to);
    EXPECT_TRUE((T.transpose() * T).isApprox(Eigen::Matrix2d::Identity(), 1e-12));
    EXPECT_TRUE((brew::usggiw::transport_matrix(to, from) * T)
                    .isApprox(Eigen::Matrix2d::Identity(), 1e-12));
}

TEST(USGGIW, PredictionPreservesUnitNorm) {
    Filter filter;
    filter.set_process_noise(1e-5 * Eigen::Matrix4d::Identity());
    const auto predicted = filter.predict(1.0, make_model(Eigen::Vector3d::UnitX(), {0.2, -0.1}));
    EXPECT_NEAR(predicted.direction().norm(), 1.0, 1e-12);
    EXPECT_EQ(predicted.covariance().rows(), 4);
}

TEST(USGGIW, TransportedConstantVelocityJacobianApproximatesStormGeodesic) {
    const auto check = [](const Eigen::Vector3d& direction,
                          const Eigen::Vector2d& velocity) {
        constexpr double dt = 16.0;
        const Eigen::Matrix4d local_cv =
            brew::usggiw::transition_jacobian(direction, velocity, dt);
        const Eigen::Matrix4d geodesic =
            finite_difference_transition_jacobian(direction, velocity, dt);
        const double relative_error = (local_cv - geodesic).norm() / geodesic.norm();
        EXPECT_LT(relative_error, 5e-5)
            << "local CV:\n" << local_cv << "\ngeodesic:\n" << geodesic;
    };

    constexpr double earth_radius_km = 6371.0;
    const double nominal_rate = 25.0 / 60.0 / earth_radius_km;
    const double stress_rate = 300.0 / 60.0 / earth_radius_km;
    check(lonlat(0.7, -0.35), nominal_rate * Eigen::Vector2d(0.8, -0.6));
    check(lonlat(-2.8, 0.1), stress_rate * Eigen::Vector2d(1.0, 0.0));
    check(lonlat(1.2, 0.65), stress_rate * Eigen::Vector2d(-0.4, 0.916515));
}

TEST(USGGIW, TransportedConstantVelocityJacobianNeglectsCurvatureCoupling) {
    const Eigen::Vector3d direction = lonlat(0.7, -0.35);
    const Eigen::Vector2d velocity(0.003, -0.002);
    constexpr double dt = 16.0;
    const auto propagated = brew::usggiw::propagate(direction, velocity, dt);
    const Eigen::Matrix2d transport =
        brew::usggiw::propagation_extent_transport(direction, propagated);

    const Eigen::Matrix4d F = brew::usggiw::transition_jacobian(
        direction, velocity, dt, propagated);

    EXPECT_TRUE((F.topLeftCorner<2, 2>().isApprox(transport, 1e-12)));
    EXPECT_TRUE((F.topRightCorner<2, 2>().isApprox(dt * transport, 1e-12)));
    EXPECT_TRUE((F.bottomLeftCorner<2, 2>().isZero(1e-12)));
    EXPECT_TRUE((F.bottomRightCorner<2, 2>().isApprox(transport, 1e-12)));
}

TEST(USGGIW, TransportedConstantVelocityJacobianAtZeroSpeed) {
    const Eigen::Matrix4d F = brew::usggiw::transition_jacobian(
        lonlat(-0.2, 0.4), Eigen::Vector2d::Zero(), 3.5);
    Eigen::Matrix4d expected = Eigen::Matrix4d::Identity();
    expected.topRightCorner<2, 2>() = 3.5 * Eigen::Matrix2d::Identity();
    EXPECT_TRUE(F.isApprox(expected, 1e-12));
}

TEST(USGGIW, VelocityUsesEastNorthTangentBasis) {
    const Eigen::Vector3d direction = lonlat(0.4, 0.3);
    const Eigen::Vector2d velocity(0.02, -0.01);
    const double dt = 5.0;
    const auto propagated = brew::usggiw::propagate(direction, velocity, dt);
    EXPECT_TRUE(brew::usggiw::local_error(direction, propagated.direction)
                    .isApprox(dt * velocity, 1e-12));
    const Eigen::Matrix2d T = brew::usggiw::transport_matrix(
        direction, propagated.direction);
    EXPECT_TRUE(propagated.velocity.isApprox(T * velocity, 1e-12));
    EXPECT_TRUE(brew::usggiw::propagation_extent_transport(direction, propagated)
                    .isApprox(T, 1e-12));
}

TEST(USGGIW, PositiveLatitudeRateMovesNorth) {
    const auto propagated = brew::usggiw::propagate(
        lonlat(-0.7, 0.2), Eigen::Vector2d(0.0, 0.03), 2.0);
    EXPECT_NEAR(propagated.direction.z(), std::sin(0.26), 1e-12);
}

TEST(USGGIW, StateResetTransportsPositionAndVelocityCoordinates) {
    const Eigen::Vector3d from = lonlat(-0.4, 0.2);
    const Eigen::Vector3d to = lonlat(0.5, -0.1);
    const Eigen::Matrix4d G = brew::usggiw::state_transport_matrix(from, to);
    const Eigen::Matrix2d T = brew::usggiw::transport_matrix(from, to);
    EXPECT_TRUE((G.topLeftCorner<2, 2>().isApprox(T, 1e-15)));
    EXPECT_TRUE((G.bottomRightCorner<2, 2>().isApprox(T, 1e-15)));
    EXPECT_TRUE((G.topRightCorner<2, 2>().isZero(1e-15)));
    EXPECT_TRUE((G.bottomLeftCorner<2, 2>().isZero(1e-15)));
}

TEST(USGGIW, PredictionCrossesDateLineContinuously) {
    const double eps = 1e-3;
    Eigen::Vector3d r(-std::cos(eps), std::sin(eps), 0.0);
    Filter filter;
    filter.set_process_noise(Eigen::Matrix4d::Zero());
    const auto predicted = filter.predict(0.02, make_model(r, {0.1, 0.0}));
    EXPECT_LT(predicted.direction().y(), 0.0);
    EXPECT_NEAR(predicted.direction().norm(), 1.0, 1e-12);
}

TEST(USGGIW, CorrectionUsesTangentInnovation) {
    Filter filter;
    filter.set_measurement_noise(1e-4 * Eigen::Matrix2d::Identity());
    auto predicted = make_model(Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    Eigen::Vector3d z = brew::usggiw::apply_position_error(
        Eigen::Vector3d::UnitX(), {0.05, -0.03});
    auto result = filter.correct(z, predicted);
    EXPECT_GT(result.distribution.direction().dot(z), predicted.direction().dot(z));
    EXPECT_TRUE(result.distribution.covariance().allFinite());
    EXPECT_GT(result.distribution.alpha(), predicted.alpha());
    EXPECT_GT(result.distribution.beta(), predicted.beta());
    EXPECT_GT(result.distribution.v(), predicted.v());
    EXPECT_FALSE(result.distribution.V().isApprox(predicted.V()));
}

TEST(USGGIW, CorrectionUpdatesAngularVelocity) {
    Filter filter;
    filter.set_measurement_noise(1e-4 * Eigen::Matrix2d::Identity());
    auto predicted = make_model(Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    predicted.covariance().topRightCorner<2, 2>() = 0.05 * Eigen::Matrix2d::Identity();
    predicted.covariance().bottomLeftCorner<2, 2>() = 0.05 * Eigen::Matrix2d::Identity();
    Eigen::Vector3d z = brew::usggiw::apply_position_error(
        Eigen::Vector3d::UnitX(), {0.04, 0.02});
    auto result = filter.correct(z, predicted);
    EXPECT_GT(result.distribution.angular_velocity().norm(), 0.0);
}

TEST(USGGIW, PointSetLikelihoodUsesScatter) {
    Filter filter;
    filter.set_measurement_noise(1e-4 * Eigen::Matrix2d::Identity());
    const auto predicted = make_model(Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    Eigen::Matrix<double, 3, 2> compact;
    compact.col(0) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.01, 0.0});
    compact.col(1) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {-0.01, 0.0});
    Eigen::Matrix<double, 3, 2> dispersed;
    dispersed.col(0) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.2, 0.0});
    dispersed.col(1) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {-0.2, 0.0});
    const auto compact_result = filter.correct(
        Eigen::Map<const Eigen::VectorXd>(compact.data(), compact.size()), predicted);
    const auto dispersed_result = filter.correct(
        Eigen::Map<const Eigen::VectorXd>(dispersed.data(), dispersed.size()), predicted);
    EXPECT_GT(compact_result.likelihood, dispersed_result.likelihood);
}

TEST(USGGIW, CorrectionTransformsCentroidAndScatterIntoExtentScale) {
    Filter filter;
    const Eigen::Matrix2d sensor_noise =
        (Eigen::Vector2d(0.002, 0.006)).asDiagonal();
    constexpr double rho = 0.4;
    filter.set_measurement_noise(sensor_noise);
    filter.set_scaling_parameter(rho);

    auto predicted = make_model(Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    Eigen::Matrix<double, 2, 3> errors;
    errors << 0.04, -0.02, 0.01,
             -0.01,  0.03, 0.02;
    Eigen::Matrix<double, 3, 3> measurements;
    for (int column = 0; column < measurements.cols(); ++column)
        measurements.col(column) = brew::usggiw::apply_position_error(
            predicted.direction(), errors.col(column));

    const Eigen::Vector2d centroid = errors.rowwise().mean();
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (int column = 0; column < errors.cols(); ++column) {
        const Eigen::Vector2d centered = errors.col(column) - centroid;
        scatter += centered * centered.transpose();
    }
    const Eigen::Matrix2d expected_extent = predicted.V() / (predicted.v() - 6.0);
    const Eigen::Matrix2d point_covariance = rho * expected_extent + sensor_noise;
    const Eigen::Matrix2d innovation_covariance =
        predicted.covariance().topLeftCorner<2, 2>()
        + point_covariance / static_cast<double>(errors.cols());
    const auto sqrt_spd = [](const Eigen::Matrix2d& matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(matrix);
        return (solver.eigenvectors()
            * solver.eigenvalues().cwiseSqrt().asDiagonal()
            * solver.eigenvectors().transpose()).eval();
    };
    const Eigen::Matrix2d sqrt_extent = sqrt_spd(expected_extent);
    const Eigen::Matrix2d inv_sqrt_point =
        sqrt_spd(point_covariance).inverse();
    const Eigen::Matrix2d inv_sqrt_innovation =
        sqrt_spd(innovation_covariance).inverse();
    const Eigen::Matrix2d transformed_scatter = sqrt_extent * inv_sqrt_point
        * scatter * inv_sqrt_point.transpose() * sqrt_extent.transpose();
    const Eigen::Matrix2d transformed_centroid = sqrt_extent * inv_sqrt_innovation
        * (centroid * centroid.transpose())
        * inv_sqrt_innovation.transpose() * sqrt_extent.transpose();

    const auto result = filter.correct(
        Eigen::Map<const Eigen::VectorXd>(measurements.data(), measurements.size()), predicted);
    const Eigen::Matrix2d reset = brew::usggiw::transport_matrix(
        predicted.direction(), result.distribution.direction());
    const Eigen::Matrix2d expected_scale = reset
        * (predicted.V() + transformed_scatter + transformed_centroid)
        * reset.transpose();

    EXPECT_TRUE(result.distribution.V().isApprox(expected_scale, 1e-12));
    EXPECT_TRUE(std::isfinite(result.likelihood));
    EXPECT_GT(result.likelihood, 0.0);
}

TEST(USGGIW, MergeIsContinuousAcrossDateLine) {
    brew::models::Mixture<Model> mix;
    const double eps = 0.01;
    mix.add_component(std::make_unique<Model>(make_model(
        {-std::cos(eps), std::sin(eps), 0.0}, {0.1, 0.0})), 0.5);
    mix.add_component(std::make_unique<Model>(make_model(
        {-std::cos(eps), -std::sin(eps), 0.0}, {0.1, 0.0})), 0.5);
    brew::usggiw::merge(mix, 25.0);
    ASSERT_EQ(mix.size(), 1);
    EXPECT_LT(mix.component(0).direction().x(), -0.99);
    EXPECT_NEAR(mix.component(0).direction().y(), 0.0, 1e-6);
}

TEST(USGGIW, MergeDirectionIsNormalizedWeightedResultant) {
    brew::models::Mixture<Model> mix;
    const Eigen::Vector3d first = brew::usggiw::apply_position_error(
        Eigen::Vector3d::UnitX(), {-0.04, 0.02});
    const Eigen::Vector3d second = brew::usggiw::apply_position_error(
        Eigen::Vector3d::UnitX(), {0.01, -0.03});
    const Eigen::Vector3d third = brew::usggiw::apply_position_error(
        Eigen::Vector3d::UnitX(), {0.05, 0.01});
    const Eigen::Vector3d expected =
        (0.2 * first + 0.3 * second + 0.5 * third).normalized();
    mix.add_component(std::make_unique<Model>(make_model(
        first, Eigen::Vector2d::Zero())), 0.2);
    mix.add_component(std::make_unique<Model>(make_model(
        second, Eigen::Vector2d::Zero())), 0.3);
    mix.add_component(std::make_unique<Model>(make_model(
        third, Eigen::Vector2d::Zero())), 0.5);

    brew::usggiw::merge(mix, 25.0);

    ASSERT_EQ(mix.size(), 1);
    EXPECT_TRUE(mix.component(0).direction().isApprox(expected, 1e-12));
}

TEST(USGGIW, MergePreservesGammaRateMeanAndExpectedExtent) {
    brew::models::Mixture<Model> mix;
    auto first = make_model(Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    first.alpha() = 4.0;
    first.beta() = 2.0;
    first.v() = 10.0;
    first.V() = 4.0 * Eigen::Matrix2d::Identity();
    auto second = make_model(Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero());
    second.alpha() = 18.0;
    second.beta() = 3.0;
    second.v() = 14.0;
    second.V() = 16.0 * Eigen::Matrix2d::Identity();
    mix.add_component(std::make_unique<Model>(first), 0.25);
    mix.add_component(std::make_unique<Model>(second), 0.75);
    brew::usggiw::merge(mix, 1.0);
    ASSERT_EQ(mix.size(), 1);
    EXPECT_NEAR(mix.component(0).alpha() / mix.component(0).beta(), 5.0, 1e-12);
    const Eigen::Matrix2d expected_extent = mix.component(0).V()
        / (mix.component(0).v() - 6.0);
    EXPECT_TRUE(expected_extent.isApprox(1.75 * Eigen::Matrix2d::Identity(), 1e-12));
}

TEST(USGGIW, GciFusesDirectionsInTangentSpace) {
    brew::models::Mixture<Model> a, b;
    a.add_component(std::make_unique<Model>(make_model(
        brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {-0.02, 0.0}), {0.0, 0.0})), 1.0);
    b.add_component(std::make_unique<Model>(make_model(
        brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.02, 0.0}), {0.0, 0.0})), 1.0);
    std::vector<const brew::models::Mixture<Model>*> sources{&a, &b};
    auto fused = brew::usggiw::gci(sources, {0.5, 0.5});
    ASSERT_EQ(fused.size(), 1);
    EXPECT_GT(fused.component(0).direction().x(), 0.999);
}

TEST(USGGIW, GciIsInvariantToSourceOrder) {
    brew::models::Mixture<Model> first, second, third;
    auto first_model = make_model(
        brew::usggiw::apply_position_error(
            Eigen::Vector3d::UnitX(), {-0.08, 0.03}),
        {0.003, -0.002});
    first_model.covariance() = Eigen::DiagonalMatrix<double, 4>(
        0.008, 0.012, 0.002, 0.003).toDenseMatrix();
    auto second_model = make_model(
        brew::usggiw::apply_position_error(
            Eigen::Vector3d::UnitX(), {0.02, -0.05}),
        {-0.001, 0.004});
    second_model.covariance() = Eigen::DiagonalMatrix<double, 4>(
        0.015, 0.009, 0.004, 0.002).toDenseMatrix();
    auto third_model = make_model(
        brew::usggiw::apply_position_error(
            Eigen::Vector3d::UnitX(), {0.11, 0.04}),
        {0.002, 0.001});
    third_model.covariance() = Eigen::DiagonalMatrix<double, 4>(
        0.006, 0.018, 0.003, 0.005).toDenseMatrix();
    first.add_component(std::make_unique<Model>(first_model), 1.0);
    second.add_component(std::make_unique<Model>(second_model), 1.0);
    third.add_component(std::make_unique<Model>(third_model), 1.0);

    std::vector<const brew::models::Mixture<Model>*> forward{
        &first, &second, &third};
    std::vector<const brew::models::Mixture<Model>*> reverse{
        &third, &second, &first};
    const auto fused_forward = brew::usggiw::gci(
        forward, {0.2, 0.3, 0.5});
    const auto fused_reverse = brew::usggiw::gci(
        reverse, {0.5, 0.3, 0.2});

    ASSERT_EQ(fused_forward.size(), 1);
    ASSERT_EQ(fused_reverse.size(), 1);
    const auto& a = fused_forward.component(0);
    const auto& b = fused_reverse.component(0);
    EXPECT_TRUE(a.direction().isApprox(b.direction(), 1e-12));
    EXPECT_TRUE(a.angular_velocity().isApprox(b.angular_velocity(), 1e-12));
    EXPECT_TRUE(a.covariance().isApprox(b.covariance(), 1e-12));
    EXPECT_TRUE(a.V().isApprox(b.V(), 1e-12));
    EXPECT_NEAR(a.alpha(), b.alpha(), 1e-12);
    EXPECT_NEAR(a.beta(), b.beta(), 1e-12);
    EXPECT_NEAR(a.v(), b.v(), 1e-12);
}

TEST(USGGIW, ComponentCorrectionUpdatesStateAndParameters) {
    Filter filter;
    filter.set_ci_weight(0.5);
    auto predicted = make_model(Eigen::Vector3d::UnitX(), {0.0, 0.0});
    auto measurement = make_model(
        brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.04, -0.02}), {0.02, -0.01});
    measurement.alpha() = 18.0;
    measurement.beta() = 3.0;
    measurement.v() = 16.0;
    measurement.V() = 2.0 * Eigen::Matrix2d::Identity();
    const auto result = filter.correct_component(measurement, predicted);
    EXPECT_GT(result.distribution.direction().dot(measurement.direction()),
              predicted.direction().dot(measurement.direction()));
    EXPECT_GT(result.distribution.angular_velocity().norm(), 0.0);
    EXPECT_NE(result.distribution.alpha(), predicted.alpha());
    EXPECT_NE(result.distribution.beta(), predicted.beta());
    EXPECT_NE(result.distribution.v(), predicted.v());
    EXPECT_FALSE(result.distribution.V().isApprox(predicted.V()));
}

TEST(USGGIW, PhdUpdatesAndMergesComponents) {
    Filter filter;
    filter.set_process_noise(1e-5 * Eigen::Matrix4d::Identity());
    filter.set_measurement_noise(1e-4 * Eigen::Matrix2d::Identity());
    brew::usggiw::PHD<> phd;
    phd.set_filter(filter.clone());
    brew::models::Mixture<Model> intensity;
    intensity.add_component(std::make_unique<Model>(make_model(
        Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero())), 0.8);
    phd.set_intensity(intensity.clone());
    phd.set_prob_detection(0.9);
    phd.set_prob_survive(0.99);
    phd.set_clutter_rate(1e-4);
    phd.set_clutter_density(1.0);
    phd.set_gate_threshold(25.0);
    phd.set_merge_threshold(25.0);
    phd.predict(0, 1.0);
    Eigen::MatrixXd measurements(3, 2);
    measurements.col(0) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.01, 0.0});
    measurements.col(1) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {-0.01, 0.0});
    phd.correct(measurements);
    phd.cleanup();
    ASSERT_GT(phd.intensity().size(), 0);
    EXPECT_NEAR(phd.intensity().component(0).direction().norm(), 1.0, 1e-12);
    EXPECT_GT(phd.intensity().component(0).alpha(), 10.0);
}

TEST(USGGIW, PhdRetainsDetectedTargetWithZeroGeneratedMeasurements) {
    Filter filter;
    brew::usggiw::PHD<> phd;
    phd.set_filter(filter.clone());
    brew::models::Mixture<Model> intensity;
    intensity.add_component(std::make_unique<Model>(make_model(
        Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero())), 0.8);
    phd.set_intensity(intensity.clone());
    phd.set_prob_detection(1.0);
    phd.correct(Eigen::MatrixXd(3, 0));
    ASSERT_EQ(phd.intensity().size(), 1);
    EXPECT_NEAR(phd.intensity().component(0).beta(), 3.0, 1e-12);
    EXPECT_NEAR(phd.intensity().weight(0), 0.8 * std::pow(2.0 / 3.0, 10.0), 1e-12);
}

TEST(USGGIW, GlmbPreservesLabelThroughMissedDetection) {
    Filter filter;
    filter.set_process_noise(1e-5 * Eigen::Matrix4d::Identity());
    brew::usggiw::GLMB<> glmb;
    glmb.set_filter(filter.clone());
    glmb.set_prob_detection(0.9);
    glmb.set_prob_survive(0.999);
    glmb.set_clutter_rate(1e-3);
    glmb.set_clutter_density(1.0);
    glmb.set_k_best(20);

    brew::models::Mixture<Model> birth;
    birth.add_component(std::make_unique<Model>(make_model(
        Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero())), 1.0);
    glmb.set_birth_model(birth.clone(), 0.99);
    glmb.predict(0, 1.0);
    Eigen::MatrixXd z(3, 1);
    z.col(0) = brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.01, 0.0});
    glmb.correct(z);
    glmb.cleanup();
    ASSERT_EQ(glmb.extracted_mixtures().back()->size(), 1u);
    const auto first_histories = glmb.track_histories();
    ASSERT_EQ(first_histories.size(), 1u);
    const int label = first_histories.begin()->first;

    Eigen::MatrixXd observers(3, 1);
    observers.col(0) = -Eigen::Vector3d::UnitX();
    glmb.set_detection_field(observers, 0.1, 0.0);
    glmb.set_birth_terms({});
    glmb.predict(1, 1.0);
    glmb.correct(Eigen::MatrixXd(3, 0));
    glmb.cleanup();
    ASSERT_EQ(glmb.extracted_mixtures().back()->size(), 1u);
    const auto second_histories = glmb.track_histories();
    ASSERT_TRUE(second_histories.contains(label));
    EXPECT_EQ(second_histories.at(label).size(), 2u);
}

TEST(USGGIW, GlmbComponentUpdateRetainsLabel) {
    Filter filter;
    filter.set_ci_weight(0.5);
    brew::usggiw::GLMB<> glmb;
    glmb.set_filter(filter.clone());
    glmb.set_prob_detection(0.95);
    glmb.set_prob_survive(0.999);
    glmb.set_clutter_rate(1e-3);
    glmb.set_clutter_density(1.0);
    glmb.set_k_best(20);
    glmb.set_gating_on(true);
    glmb.set_gate_threshold(25.0);

    brew::models::Mixture<Model> birth;
    birth.add_component(std::make_unique<Model>(make_model(
        Eigen::Vector3d::UnitX(), Eigen::Vector2d::Zero())), 1.0);
    glmb.set_birth_model(birth.clone(), 0.99);
    glmb.predict(0, 1.0);
    brew::models::Mixture<Model> report;
    report.add_component(std::make_unique<Model>(make_model(
        brew::usggiw::apply_position_error(Eigen::Vector3d::UnitX(), {0.02, -0.01}), {0.01, 0.0})), 1.0);
    glmb.set_measurement_components(report);
    Eigen::MatrixXd z(3, 1);
    z.col(0) = report.component(0).direction();
    glmb.correct(z);
    glmb.cleanup();
    ASSERT_EQ(glmb.track_histories().size(), 1u);
    EXPECT_EQ(glmb.extracted_mixtures().back()->size(), 1u);
    EXPECT_GT(glmb.extracted_mixtures().back()->component(0).angular_velocity().norm(), 0.0);
}

TEST(USGGIW, GeographicPolesAreSingular) {
    EXPECT_THROW(brew::usggiw::east_axis(Eigen::Vector3d::UnitZ()), std::domain_error);
    EXPECT_THROW(brew::usggiw::east_axis(-Eigen::Vector3d::UnitZ()), std::domain_error);
}
