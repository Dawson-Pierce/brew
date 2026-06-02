#include <gtest/gtest.h>

// Compile-smoke test for the per-package CONCRETE RFS filters of all NON-gaussian
// packages (gaussian has its own test_gaussian_concrete.cpp). Each package's RFS
// are concrete classes in namespace brew::<pkg>, templated only on data type +
// size/shape (and extent De for the GGIW family), never on the model. This forces
// full template + vtable emission of every concrete RFS through its umbrella,
// catching de-templating regressions at compile time; clone() exercises the
// RFSBase override on a default-constructed filter.
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

// Instantiate the 6 concrete RFS of one package (MBMBase is abstract -> only via
// the MBM/PMBM inheritance static_asserts).
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
