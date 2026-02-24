#include <gtest/gtest.h>
#include "brew/plot_utils/math_utils.hpp"
#include <numbers>
#include <cmath>

using namespace brew::plot_utils;

TEST(Chi2Inv, ExactDof2) {
    // chi2inv(0.95, 2) = -2*ln(0.05) ≈ 5.9915
    double result = chi2inv(0.95, 2);
    EXPECT_NEAR(result, 5.9915, 0.001);

    // chi2inv(0.99, 2) = -2*ln(0.01) ≈ 9.2103
    result = chi2inv(0.99, 2);
    EXPECT_NEAR(result, 9.2103, 0.001);
}

TEST(Chi2Inv, GeneralDof) {
    // chi2inv(0.95, 1) ≈ 3.8415
    double result = chi2inv(0.95, 1);
    EXPECT_NEAR(result, 3.8415, 0.1);

    // chi2inv(0.95, 3) ≈ 7.8147
    result = chi2inv(0.95, 3);
    EXPECT_NEAR(result, 7.8147, 0.2);

    // chi2inv(0.95, 10) ≈ 18.307
    result = chi2inv(0.95, 10);
    EXPECT_NEAR(result, 18.307, 0.5);
}

TEST(Chi2Inv, BoundaryValues) {
    EXPECT_DOUBLE_EQ(chi2inv(0.0, 2), 0.0);
    EXPECT_EQ(chi2inv(1.0, 2), std::numeric_limits<double>::infinity());
}

TEST(NormPdf, StandardNormal) {
    // normpdf(0, 0, 1) = 1/sqrt(2*pi) ≈ 0.3989
    EXPECT_NEAR(normpdf(0.0), 0.3989, 0.001);

    // normpdf(1, 0, 1) ≈ 0.2420
    EXPECT_NEAR(normpdf(1.0), 0.2420, 0.001);
}

TEST(NormPdf, NonStandard) {
    // normpdf(5, 5, 2) = 1/(2*sqrt(2*pi)) ≈ 0.1995
    EXPECT_NEAR(normpdf(5.0, 5.0, 2.0), 0.1995, 0.001);
}

TEST(Linspace, BasicTests) {
    auto v = linspace(0.0, 1.0, 5);
    ASSERT_EQ(v.size(), 5);
    EXPECT_DOUBLE_EQ(v[0], 0.0);
    EXPECT_DOUBLE_EQ(v[4], 1.0);
    EXPECT_NEAR(v[2], 0.5, 1e-10);
}

TEST(Linspace, SinglePoint) {
    auto v = linspace(3.0, 7.0, 1);
    ASSERT_EQ(v.size(), 1);
    EXPECT_DOUBLE_EQ(v[0], 7.0);
}

TEST(GenerateEllipsePoints, IdentityCircle) {
    Eigen::VectorXd mean(2);
    mean << 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);
    std::vector<int> inds = {0, 1};

    auto pts = generate_ellipse_points(mean, cov, inds, 1.0, 100);
    ASSERT_EQ(pts.rows(), 100);
    ASSERT_EQ(pts.cols(), 2);

    // All points should be approximately on the unit circle
    for (int i = 0; i < pts.rows(); ++i) {
        double r = std::sqrt(pts(i, 0) * pts(i, 0) + pts(i, 1) * pts(i, 1));
        EXPECT_NEAR(r, 1.0, 0.05);
    }
}

TEST(GenerateEllipsePoints, ScaledEllipse) {
    Eigen::VectorXd mean(2);
    mean << 5.0, 3.0;
    Eigen::MatrixXd cov(2, 2);
    cov << 4.0, 0.0,
           0.0, 1.0;
    std::vector<int> inds = {0, 1};

    auto pts = generate_ellipse_points(mean, cov, inds, 1.0, 100);

    // Check center offset
    double avg_x = 0, avg_y = 0;
    for (int i = 0; i < pts.rows(); ++i) {
        avg_x += pts(i, 0);
        avg_y += pts(i, 1);
    }
    avg_x /= pts.rows();
    avg_y /= pts.rows();
    EXPECT_NEAR(avg_x, 5.0, 0.1);
    EXPECT_NEAR(avg_y, 3.0, 0.1);
}

TEST(GenerateEllipsoidMesh, Identity) {
    Eigen::VectorXd mean(3);
    mean << 0.0, 0.0, 0.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(3, 3);
    std::vector<int> inds = {0, 1, 2};

    auto mesh = generate_ellipsoid_mesh(mean, cov, inds, 1.0, 10);
    EXPECT_EQ(mesh.X.rows(), 11);
    EXPECT_EQ(mesh.X.cols(), 11);

    // All points should be approximately on the unit sphere
    for (int i = 0; i < mesh.X.rows(); ++i) {
        for (int j = 0; j < mesh.X.cols(); ++j) {
            double r = std::sqrt(mesh.X(i, j) * mesh.X(i, j) +
                                 mesh.Y(i, j) * mesh.Y(i, j) +
                                 mesh.Z(i, j) * mesh.Z(i, j));
            EXPECT_NEAR(r, 1.0, 0.05);
        }
    }
}

TEST(UnitSphere, Shape) {
    auto mesh = unit_sphere(20);
    EXPECT_EQ(mesh.X.rows(), 21);
    EXPECT_EQ(mesh.X.cols(), 21);
}

TEST(EigenConversion, VectorConversion) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    auto sv = eigen_to_vec(v);
    ASSERT_EQ(sv.size(), 3);
    EXPECT_DOUBLE_EQ(sv[0], 1.0);
    EXPECT_DOUBLE_EQ(sv[2], 3.0);
}

TEST(EigenConversion, MatrixConversion) {
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3,
         4, 5, 6;
    auto sm = eigen_to_mat(m);
    ASSERT_EQ(sm.size(), 2);
    ASSERT_EQ(sm[0].size(), 3);
    EXPECT_DOUBLE_EQ(sm[1][2], 6.0);
}

