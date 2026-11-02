#include <gtest/gtest.h>
#include "brew/utils/fusion/arithmetic_average.hpp"
#include "brew/utils/fusion/gaussian_gci.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/utils/fusion/ggiw_gci.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/utils/fusion/iggiw_gci.hpp"
#include "brew/models/iggiw.hpp"
#include "brew/utils/fusion/ggiw_orientation_gci.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include "brew/utils/fusion/template_pose_gci.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/utils/fusion/trajectory_gaussian_gci.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/utils/fusion/trajectory_ggiw_gci.hpp"
#include "brew/models/trajectory_ggiw.hpp"
#include "brew/utils/fusion/trajectory_template_pose_gci.hpp"
#include "brew/models/trajectory_template_pose.hpp"
#include "brew/models/mixture.hpp"

using namespace brew;
using G = brew::models::Gaussian<>;
using Mix = brew::models::Mixture<G>;
using GG = brew::models::GGIW<>;
using GGMix = brew::models::Mixture<GG>;

static Mix one(const Eigen::Vector2d& m, const Eigen::Matrix2d& P, double w = 1.0) {
    Mix mix;
    mix.add_component(std::make_unique<G>(m, P), w);
    return mix;
}

TEST(Fusion, ArithmeticAverageUnionAndScale) {
    Mix a = one({0, 0}, Eigen::Matrix2d::Identity());
    Mix b = one({5, 5}, Eigen::Matrix2d::Identity());
    auto f = utils::fusion::arithmetic_average<G>({&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 2u);
    EXPECT_NEAR(f.weight(0), 0.5, 1e-12);
    EXPECT_NEAR(f.weight(1), 0.5, 1e-12);
}

TEST(Fusion, ArithmeticAverageNormalizesValidSourceWeights) {
    Mix a = one({0, 0}, Eigen::Matrix2d::Identity());
    Mix b = one({5, 5}, Eigen::Matrix2d::Identity());
    auto f = utils::fusion::arithmetic_average<G>({&a, nullptr, &b}, {2.0, 100.0, 1.0});
    ASSERT_EQ(f.size(), 2u);
    EXPECT_NEAR(f.weight(0), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.weight(1), 1.0 / 3.0, 1e-12);
}

TEST(Fusion, GciIdenticalIsIdentity) {
    Eigen::Vector2d m(1, 2);
    Eigen::Matrix2d P;
    P << 2, 0.3, 0.3, 1;
    Mix a = one(m, P), b = one(m, P);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).mean().isApprox(m, 1e-9));
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-9));
    EXPECT_NEAR(f.weight(0), 1.0, 1e-9);
}

TEST(Fusion, GciRejectsDegenerateSourceWeights) {
    Mix a = one({0, 0}, Eigen::Matrix2d::Identity());
    Mix b = one({1, 0}, Eigen::Matrix2d::Identity());
    EXPECT_THROW((utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.0, 0.0})), std::invalid_argument);
    EXPECT_THROW((utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {1.0, -0.1})), std::invalid_argument);
}

TEST(Fusion, GciSameCovAveragesMean) {
    Eigen::Vector2d m1(0, 0), m2(4, 2);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 3.0;
    Mix a = one(m1, P), b = one(m2, P);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).mean().isApprox((m1 + m2) * 0.5, 1e-9));
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-9));
}

TEST(Fusion, GciDiffCovInformationForm) {
    Eigen::Vector2d z = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P1 = Eigen::Matrix2d::Identity() * 2.0, P2;
    P2 << 1, 0, 0, 4;
    Mix a = one(z, P1), b = one(z, P2);
    const double w = 0.5;
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {w, 1 - w});
    Eigen::Matrix2d expected_info = w * P1.inverse() + (1 - w) * P2.inverse();
    EXPECT_TRUE(f.component(0).covariance().inverse().isApprox(expected_info, 1e-9));
}

static GGMix ggiw_one(double a, double b, const Eigen::Vector2d& m, const Eigen::Matrix2d& P,
                      double v, const Eigen::Matrix2d& V, double w = 1.0) {
    GGMix mix;
    mix.add_component(std::make_unique<GG>(a, b, m, P, v, V), w);
    return mix;
}

TEST(Fusion, GgiwGciIdenticalIsIdentity) {
    Eigen::Vector2d m(1, 2);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 2.0;
    Eigen::Matrix2d V = Eigen::Matrix2d::Identity() * 3.0;
    GGMix a = ggiw_one(5, 2, m, P, 10, V), b = ggiw_one(5, 2, m, P, 10, V);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_NEAR(f.component(0).alpha(), 5.0, 1e-9);
    EXPECT_NEAR(f.component(0).beta(), 2.0, 1e-9);
    EXPECT_NEAR(f.component(0).v(), 10.0, 1e-9);
    EXPECT_TRUE(f.component(0).V().isApprox(V, 1e-9));
    EXPECT_TRUE(f.component(0).mean().isApprox(m, 1e-9));
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-9));
    EXPECT_NEAR(f.weight(0), 1.0, 1e-9);
}

TEST(Fusion, GgiwGciNaturalParamPooling) {
    Eigen::Vector2d m1(0, 0), m2(2, 1);
    Eigen::Matrix2d P1 = Eigen::Matrix2d::Identity() * 2.0, P2;
    P2 << 1, 0, 0, 4;
    Eigen::Matrix2d V1 = Eigen::Matrix2d::Identity() * 3.0, V2 = Eigen::Matrix2d::Identity() * 6.0;
    const double w1 = 0.3, w2 = 0.7;
    GGMix a = ggiw_one(4, 2, m1, P1, 8, V1), b = ggiw_one(6, 4, m2, P2, 12, V2);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {w1, w2});
    ASSERT_EQ(f.size(), 1u);
    const auto& g = f.component(0);
    EXPECT_NEAR(g.alpha(), w1 * 4 + w2 * 6, 1e-9);
    EXPECT_NEAR(g.beta(), w1 * 2 + w2 * 4, 1e-9);
    EXPECT_NEAR(g.v(), w1 * 8 + w2 * 12, 1e-9);
    EXPECT_TRUE(g.V().isApprox(w1 * V1 + w2 * V2, 1e-9));
    EXPECT_TRUE(g.covariance().inverse().isApprox(w1 * P1.inverse() + w2 * P2.inverse(), 1e-9));
    Eigen::Vector2d expected_m = g.covariance() * (w1 * P1.inverse() * m1 + w2 * P2.inverse() * m2);
    EXPECT_TRUE(g.mean().isApprox(expected_m, 1e-9));
}

TEST(Fusion, IggiwGciIdentity) {
    using IG = brew::models::IGGIW<>;
    Eigen::Vector2d m(1, 2);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 2.0;
    Eigen::Matrix2d V = Eigen::Matrix2d::Identity() * 3.0;
    brew::models::Mixture<IG> a, b;
    a.add_component(std::make_unique<IG>(5.0, 2.0, m, P, 10.0, V), 1.0);
    b.add_component(std::make_unique<IG>(5.0, 2.0, m, P, 10.0, V), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_NEAR(f.component(0).alpha(), 5.0, 1e-9);
    EXPECT_NEAR(f.component(0).v(), 10.0, 1e-9);
    EXPECT_TRUE(f.component(0).mean().isApprox(m, 1e-9));
}

TEST(Fusion, GgiwOrientationGciDerivesBasis) {
    using GO = brew::models::GGIWOrientation<>;
    Eigen::Vector2d m(0, 0);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 2.0;
    Eigen::Matrix2d V;
    V << 5, 1, 1, 2;
    brew::models::Mixture<GO> a, b;
    a.add_component(std::make_unique<GO>(5.0, 2.0, m, P, 10.0, V), 1.0);
    b.add_component(std::make_unique<GO>(5.0, 2.0, m, P, 10.0, V), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).V().isApprox(V, 1e-9));
    EXPECT_TRUE(f.component(0).has_eigenvalues());
}

static Eigen::Matrix3d Rz(double a) {
    Eigen::Matrix3d R;
    R << std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1;
    return R;
}

TEST(Fusion, TemplatePoseGciAugmentedManifoldCi) {
    using TP = brew::models::TemplatePose<>;
    Eigen::Vector3d t1(0, 0, 0), t2(2, 0, 0);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(6, 6) * 2.0;  // [trans(3); so(3) tangent(3)]
    std::vector<int> pos{0, 1, 2};
    brew::models::Mixture<TP> a, b;
    a.add_component(std::make_unique<TP>(t1, P, 7, pos, Rz(0.0)), 1.0);
    b.add_component(std::make_unique<TP>(t2, P, 7, pos, Rz(M_PI / 2)), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).mean().isApprox((t1 + t2) * 0.5, 1e-7));
    EXPECT_TRUE(f.component(0).rotation().isApprox(Rz(M_PI / 4), 1e-7));
    EXPECT_EQ(f.component(0).covariance().rows(), 6);
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-7));  // equal P -> fused P == P
    EXPECT_EQ(f.component(0).template_id(), 7);
}

TEST(Fusion, TemplatePoseGciIdentity) {
    using TP = brew::models::TemplatePose<>;
    Eigen::Vector3d t(1, 2, 3);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(6, 6) * 1.5;
    std::vector<int> pos{0, 1, 2};
    brew::models::Mixture<TP> a, b;
    a.add_component(std::make_unique<TP>(t, P, 3, pos, Rz(0.4)), 1.0);
    b.add_component(std::make_unique<TP>(t, P, 3, pos, Rz(0.4)), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).mean().isApprox(t, 1e-9));
    EXPECT_TRUE(f.component(0).rotation().isApprox(Rz(0.4), 1e-7));
    EXPECT_TRUE(f.component(0).covariance().isApprox(P, 1e-7));
}

TEST(Fusion, TrajectoryGaussianGciStackedCi) {
    using IG = brew::models::Gaussian<>;
    using TG = brew::models::TrajectoryGaussian<>;
    Eigen::Vector2d m1(0, 0), m2(4, 2);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 3.0;
    brew::models::Mixture<TG> a, b;
    a.add_component(std::make_unique<TG>(2, IG(m1, P), 5), 1.0);
    b.add_component(std::make_unique<TG>(2, IG(m2, P), 5), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    const auto& t = f.component(0);
    EXPECT_TRUE(t.mean().head(2).isApprox((m1 + m2) * 0.5, 1e-9));
    EXPECT_TRUE(t.covariance().topLeftCorner(2, 2).isApprox(P, 1e-9));
    EXPECT_TRUE(t.history_at(0).mean().isApprox((m1 + m2) * 0.5, 1e-9));
}

TEST(Fusion, TrajectoryGaussianGciSkipsMismatchedWindow) {
    using IG = brew::models::Gaussian<>;
    using TG = brew::models::TrajectoryGaussian<>;
    Eigen::Vector2d m(0, 0);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity();
    brew::models::Mixture<TG> a, b;
    a.add_component(std::make_unique<TG>(2, IG(m, P), 5), 1.0);           // state_dim 2
    b.add_component(std::make_unique<TG>(3, IG(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()), 5), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    EXPECT_EQ(f.size(), 0u);  // different stacked size -> no fusable tuple
}

TEST(Fusion, TrajectoryGgiwGciPoolsInner) {
    using IG = brew::models::GGIW<>;
    using TG = brew::models::TrajectoryGGIW<>;
    Eigen::Vector2d m1(0, 0), m2(4, 2);
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * 3.0;
    Eigen::Matrix2d V = Eigen::Matrix2d::Identity() * 5.0;
    brew::models::Mixture<TG> a, b;
    a.add_component(std::make_unique<TG>(2, IG(5, 2, m1, P, 10, V), 5), 1.0);
    b.add_component(std::make_unique<TG>(2, IG(7, 4, m2, P, 12, V), 5), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    const auto& t = f.component(0);
    EXPECT_TRUE(t.mean().head(2).isApprox((m1 + m2) * 0.5, 1e-9));
    EXPECT_NEAR(t.history_at(0).alpha(), 6.0, 1e-9);
    EXPECT_NEAR(t.history_at(0).beta(), 3.0, 1e-9);
    EXPECT_NEAR(t.history_at(0).v(), 11.0, 1e-9);
    EXPECT_TRUE(t.history_at(0).V().isApprox(V, 1e-9));
    EXPECT_TRUE(t.history_at(0).mean().isApprox((m1 + m2) * 0.5, 1e-9));
}

TEST(Fusion, TrajectoryTemplatePoseGciSo3) {
    using IT = brew::models::TemplatePose<>;
    using TG = brew::models::TrajectoryTemplatePose<>;
    Eigen::Vector3d m(0, 0, 0);
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 2.0;
    std::vector<int> pos{0, 1, 2};
    brew::models::Mixture<TG> a, b;
    a.add_component(std::make_unique<TG>(3, IT(m, P, 5, pos, Rz(0.0)), 5), 1.0);
    b.add_component(std::make_unique<TG>(3, IT(m, P, 5, pos, Rz(M_PI / 2)), 5), 1.0);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    ASSERT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.component(0).history_at(0).rotation().isApprox(Rz(M_PI / 4), 1e-7));
}

TEST(Fusion, GciMixtureCombinatorialNormalized) {
    Eigen::Matrix2d P = Eigen::Matrix2d::Identity();
    Mix a, b;
    a.add_component(std::make_unique<G>(Eigen::Vector2d(0, 0), P), 0.5);
    a.add_component(std::make_unique<G>(Eigen::Vector2d(1, 0), P), 0.5);
    b.add_component(std::make_unique<G>(Eigen::Vector2d(0, 1), P), 0.5);
    b.add_component(std::make_unique<G>(Eigen::Vector2d(1, 1), P), 0.5);
    auto f = utils::fusion::gci(std::vector<const decltype(a)*>{&a, &b}, {0.5, 0.5});
    EXPECT_EQ(f.size(), 4u);
    double sw = 0;
    for (std::size_t i = 0; i < f.size(); ++i) sw += f.weight(i);
    EXPECT_NEAR(sw, 1.0, 1e-9);
}
