#include <gtest/gtest.h>

#include "brew/template_matching/pca_icp.hpp"
#include "brew/template_matching/point_to_point_icp.hpp"
#include "brew/template_matching/point_to_plane_icp.hpp"
#include "brew/template_matching/so3.hpp"
#include "brew/template_matching/template_library.hpp"

#include <cmath>
#include <random>

using namespace brew::template_matching;

TEST(Icp, PartialPlaneDoesNotInventMotionAlongWeaklyObservedDirections) {
    Eigen::MatrixXd points(3, 100), normals(3, 100);
    for (int i = 0; i < 100; ++i) {
        points.col(i) = Eigen::Vector3d(.3 * (i % 10), .3 * (i / 10), 0.);
        normals.col(i) = Eigen::Vector3d(1e-6 * std::sin(i), 1e-6 * std::cos(i), 1.).normalized();
    }
    const Eigen::Vector3d position(100., 20., 10.);
    Eigen::MatrixXd observed = points.colwise() + position;
    for (int i = 0; i < 100; ++i) observed(2, i) += .005 * std::sin(i);
    PointToPlaneIcp icp;
    IcpParams params;
    params.max_iterations = 100;
    params.max_correspondence_dist = 8.;
    params.tolerance = 1e-12;
    params.observability_threshold = .1;
    icp.set_params(params);
    const Eigen::Vector3d initial = position + Eigen::Vector3d(.03, -.02, .3);
    const auto result = icp.align(PointCloud(points, normals), PointCloud(observed),
                                  Eigen::Matrix3d::Identity(), initial);
    EXPECT_NEAR(result.translation.z(), position.z(), .01);
    EXPECT_LT((result.translation.head<2>() - initial.head<2>()).norm(), .01);
    EXPECT_LT(so3::log(result.rotation).norm(), .01);
}

TEST(Icp, PartialPointToPlaneAlignmentIsIndependentOfCoordinateOrigin) {
    std::mt19937 random(29);
    std::normal_distribution<double> normal;
    Eigen::MatrixXd points(3, 800);
    for (int i = 0; i < points.cols(); ++i) {
        Eigen::Vector3d direction(normal(random), normal(random), normal(random));
        points.col(i) = Eigen::Vector3d(3., 2., 1.).asDiagonal() * direction.normalized();
    }
    const PointCloud source(points);
    const Eigen::Matrix3d rotation = so3::exp(Eigen::Vector3d(.2, -.1, .05));
    const Eigen::Matrix3d initial = so3::exp(Eigen::Vector3d(.15, -.1, .12)) * rotation;
    PointToPlaneIcp icp;
    IcpParams params;
    params.max_iterations = 100;
    params.tolerance = 1e-12;
    params.trim_fraction = .8;
    params.max_correspondence_dist = 8.;
    icp.set_params(params);
    for (double shift : {0., 10000.}) {
        const Eigen::Vector3d position(100. + shift, 20. - shift, 10.);
        Eigen::MatrixXd observed(3, 800);
        int count = 0;
        for (int i = 0; i < points.cols(); ++i) {
            if (points(0, i) < -.5) observed.col(count++) = rotation * points.col(i) + position;
        }
        observed.conservativeResize(3, count);
        const auto result = icp.align(source, PointCloud(observed), initial,
                                     position + Eigen::Vector3d(.5, -.4, .3));
        EXPECT_LT((result.translation - position).norm(), 1e-5);
        EXPECT_LT(so3::log(Eigen::Matrix3d(result.rotation * rotation.transpose())).norm(), 1e-5);
    }
}

TEST(Icp, LibraryAndClonePreserveSuppliedSurfaceNormals) {
    Eigen::MatrixXd points(3, 3), normals(3, 3);
    points << 0., 1., 0., 0., 0., 1., 0., 0., 0.;
    normals << 2., 0., 0., 0., 3., 0., 0., 0., 4.;
    TemplateLibrary library;
    const int id = library.add(PointCloud(points, normals));
    const auto cloud = library.get(id);
    EXPECT_TRUE(cloud->normals(3).isApprox(Eigen::Matrix3d::Identity()));
    const auto clone = cloud->clone();
    EXPECT_TRUE(clone->normals(2).isApprox(Eigen::Matrix3d::Identity()));
    EXPECT_TRUE(cloud->points().isApprox(points));
    EXPECT_TRUE(cloud->normals(3).isApprox(Eigen::Matrix3d::Identity()));
}

TEST(Icp, ReverseCorrespondenceLikelihoodUsesCanonicalIndices) {
    Eigen::MatrixXd source(3, 3);
    source << 0.0, 10.0, 20.0,
              0.0,  0.0,  0.0,
              0.0,  0.0,  0.0;
    Eigen::MatrixXd target(3, 2);
    target << 0.0, 20.0,
              0.0,  0.0,
              0.0,  0.0;

    PointToPointIcp icp;
    const double value = icp.log_likelihood(
        PointCloud(source), PointCloud(target),
        Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    EXPECT_NEAR(value, -3.0 * std::log(2.0 * M_PI), 1e-12);
}

TEST(Icp, TargetCentroidInitializationAccountsForSourceCentroid) {
    Eigen::MatrixXd source(3, 3);
    source << -1.0, 0.0, 1.0,
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0;
    Eigen::MatrixXd target = source.colwise() + Eigen::Vector3d(5.0, -2.0, 1.0);

    PointToPointIcp icp;
    IcpParams params;
    params.use_target_centroid_init = true;
    params.max_iterations = 1;
    icp.set_params(params);
    const auto result = icp.align(PointCloud(source), PointCloud(target));
    EXPECT_TRUE(result.translation.isApprox(Eigen::Vector3d(5.0, -2.0, 1.0), 1e-12));
    EXPECT_NEAR(result.rmse, 0.0, 1e-12);
}

TEST(Icp, RejectsInvalidLikelihoodVariance) {
    PointToPointIcp icp;
    IcpParams params;
    params.sigma_sq = 0.0;
    EXPECT_THROW(icp.set_params(params), std::invalid_argument);
}

TEST(Icp, PcaClonePreservesConfigurationWithoutDummyCloud) {
    Eigen::MatrixXd source(3, 3);
    source << -1.0, 0.0, 2.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0;
    auto inner = std::make_unique<PointToPointIcp>();
    PcaIcp pca(std::move(inner), PointCloud(source));
    IcpParams params;
    params.sigma_sq = 2.5;
    pca.set_params(params);
    const auto clone = pca.clone();
    EXPECT_DOUBLE_EQ(clone->params().sigma_sq, 2.5);
}

TEST(Icp, CachedNearestNeighborMatchesBruteForce) {
    Eigen::MatrixXd points = Eigen::MatrixXd::Random(3, 257);
    PointCloud cloud(points);
    for (int i = 0; i < 40; ++i) {
        const Eigen::Vector3d query = Eigen::Vector3d::Random();
        Eigen::Index expected = -1;
        const double expected_distance =
            (points.colwise() - query).colwise().squaredNorm().minCoeff(&expected);
        const auto [actual, actual_distance] = cloud.nearest(query);
        EXPECT_EQ(actual, expected);
        EXPECT_NEAR(actual_distance, expected_distance, 1e-12);
    }

    cloud.points().col(0) = Eigen::Vector3d(10.0, 10.0, 10.0);
    const auto [updated, distance] = cloud.nearest(Eigen::Vector3d(10.0, 10.0, 10.0));
    EXPECT_EQ(updated, 0);
    EXPECT_NEAR(distance, 0.0, 1e-12);
}
