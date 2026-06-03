// Compile-smoke + clone for the concrete RFS of every non-gaussian package.

#include <gtest/gtest.h>
#include "brew/ggiw/ggiw.hpp"
#include "brew/ggiw_orientation/ggiw_orientation.hpp"
#include "brew/iggiw/iggiw.hpp"
#include "brew/template_pose/template_pose.hpp"
#include "brew/trajectory_gaussian/trajectory_gaussian.hpp"
#include "brew/trajectory_ggiw/trajectory_ggiw.hpp"
#include "brew/trajectory_ggiw_orientation/trajectory_ggiw_orientation.hpp"
#include "brew/trajectory_iggiw/trajectory_iggiw.hpp"
#include "brew/trajectory_template_pose/trajectory_template_pose.hpp"

using namespace brew;

template <typename Rfs>
static void instantiate_and_clone() {
    Rfs rfs;
    auto cloned = rfs.clone();
    EXPECT_NE(cloned, nullptr);
    static_assert(std::is_base_of_v<multi_target::RFSBase, Rfs>,
                  "concrete package RFS must derive from multi_target::RFSBase");
}

#define SMOKE_PACKAGE(NS)                                  \
    instantiate_and_clone<NS::PHD<>>();                    \
    instantiate_and_clone<NS::CPHD<>>();                   \
    instantiate_and_clone<NS::GLMB<>>();                   \
    instantiate_and_clone<NS::JGLMB<>>();                  \
    instantiate_and_clone<NS::MBM<>>();                    \
    instantiate_and_clone<NS::PMBM<>>();                   \
    static_assert(std::is_base_of_v<NS::GLMB<>, NS::JGLMB<>>);   \
    static_assert(std::is_base_of_v<NS::MBMBase<>, NS::MBM<>>);  \
    static_assert(std::is_base_of_v<NS::MBMBase<>, NS::PMBM<>>)

TEST(PackagesConcreteRFS, Ggiw)                     { SMOKE_PACKAGE(ggiw); }
TEST(PackagesConcreteRFS, GgiwOrientation)          { SMOKE_PACKAGE(ggiw_orientation); }
TEST(PackagesConcreteRFS, Iggiw)                    { SMOKE_PACKAGE(iggiw); }
TEST(PackagesConcreteRFS, TemplatePose)             { SMOKE_PACKAGE(template_pose); }
TEST(PackagesConcreteRFS, TrajectoryGaussian)       { SMOKE_PACKAGE(trajectory_gaussian); }
TEST(PackagesConcreteRFS, TrajectoryGgiw)           { SMOKE_PACKAGE(trajectory_ggiw); }
TEST(PackagesConcreteRFS, TrajectoryGgiwOrient)     { SMOKE_PACKAGE(trajectory_ggiw_orientation); }
TEST(PackagesConcreteRFS, TrajectoryIggiw)          { SMOKE_PACKAGE(trajectory_iggiw); }
TEST(PackagesConcreteRFS, TrajectoryTemplatePose)   { SMOKE_PACKAGE(trajectory_template_pose); }
