// Instantiate and clone the generic RFS algorithms with each density.

#include <gtest/gtest.h>
#include "brew/models/ggiw.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include "brew/multiObjectTrackers/mbm_base.hpp"
#include "brew/multiObjectTrackers/cphd.hpp"
#include "brew/multiObjectTrackers/glmb.hpp"
#include "brew/multiObjectTrackers/jglmb.hpp"
#include "brew/multiObjectTrackers/mbm.hpp"
#include "brew/multiObjectTrackers/pmbm.hpp"
#include "brew/models/merge/ggiw.hpp"
#include "brew/utils/fusion/ggiw_gci.hpp"
#include "brew/utils/fusion/arithmetic_average.hpp"
#include "brew/utils/fusion/prune.hpp"
#include "brew/utils/fusion/cap.hpp"
#include "brew/utils/clustering/dbscan.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include "brew/filters/ggiw_orientation_ekf.hpp"
#include "brew/models/merge/ggiw_orientation.hpp"
#include "brew/utils/fusion/ggiw_orientation_gci.hpp"
#include "brew/models/iggiw.hpp"
#include "brew/filters/iggiw_ekf.hpp"
#include "brew/models/merge/iggiw.hpp"
#include "brew/utils/fusion/iggiw_gci.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/filters/tm_ekf.hpp"
#include "brew/filters/imm_tm_ekf.hpp"
#include "brew/models/pointing_firing_model.hpp"
#include "brew/filters/intent_filter.hpp"
#include "brew/models/merge/template_pose.hpp"
#include "brew/utils/fusion/template_pose_gci.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/filters/trajectory_gaussian_ekf.hpp"
#include "brew/models/merge/trajectory_gaussian.hpp"
#include "brew/utils/fusion/trajectory_gaussian_gci.hpp"
#include "brew/models/trajectory_ggiw.hpp"
#include "brew/filters/trajectory_ggiw_ekf.hpp"
#include "brew/models/merge/trajectory_ggiw.hpp"
#include "brew/utils/fusion/trajectory_ggiw_gci.hpp"
#include "brew/models/trajectory_ggiw_orientation.hpp"
#include "brew/filters/trajectory_ggiw_orientation_ekf.hpp"
#include "brew/models/merge/trajectory_ggiw_orientation.hpp"
#include "brew/utils/fusion/trajectory_ggiw_orientation_gci.hpp"
#include "brew/models/trajectory_iggiw.hpp"
#include "brew/filters/trajectory_iggiw_ekf.hpp"
#include "brew/models/merge/trajectory_iggiw.hpp"
#include "brew/utils/fusion/trajectory_iggiw_gci.hpp"
#include "brew/models/trajectory_template_pose.hpp"
#include "brew/filters/trajectory_tm_ekf.hpp"
#include "brew/models/merge/trajectory_template_pose.hpp"
#include "brew/utils/fusion/trajectory_template_pose_gci.hpp"

using namespace brew;

template <typename Rfs>
static void instantiate_and_clone() {
    Rfs rfs;
    auto cloned = rfs.clone();
    EXPECT_NE(cloned, nullptr);
    static_assert(std::is_base_of_v<multiObjectTrackers::RFSBase, Rfs>,
                  "RFS specialization must derive from multiObjectTrackers::RFSBase");
}

template<class Density>
static void check_density() {
    instantiate_and_clone<multiObjectTrackers::PHD<Density>>();
    instantiate_and_clone<multiObjectTrackers::CPHD<Density>>();
    instantiate_and_clone<multiObjectTrackers::GLMB<Density>>();
    instantiate_and_clone<multiObjectTrackers::JGLMB<Density>>();
    instantiate_and_clone<multiObjectTrackers::MBM<Density>>();
    instantiate_and_clone<multiObjectTrackers::PMBM<Density>>();
    static_assert(std::is_base_of_v<multiObjectTrackers::GLMB<Density>, multiObjectTrackers::JGLMB<Density>>);
    static_assert(std::is_base_of_v<multiObjectTrackers::MBMBase<Density>, multiObjectTrackers::MBM<Density>>);
    static_assert(std::is_base_of_v<multiObjectTrackers::MBMBase<Density>, multiObjectTrackers::PMBM<Density>>);
}

TEST(DensityRFS, GGIW) { check_density<models::GGIW<>>(); }
TEST(DensityRFS, GGIWOrientation) { check_density<models::GGIWOrientation<>>(); }
TEST(DensityRFS, IGGIW) { check_density<models::IGGIW<>>(); }
TEST(DensityRFS, TemplatePose) { check_density<models::TemplatePose<>>(); }
TEST(DensityRFS, TrajectoryGaussian) { check_density<models::TrajectoryGaussian<>>(); }
TEST(DensityRFS, TrajectoryGGIW) { check_density<models::TrajectoryGGIW<>>(); }
TEST(DensityRFS, TrajectoryGGIWOrientation) { check_density<models::TrajectoryGGIWOrientation<>>(); }
TEST(DensityRFS, TrajectoryIGGIW) { check_density<models::TrajectoryIGGIW<>>(); }
TEST(DensityRFS, TrajectoryTemplatePose) { check_density<models::TrajectoryTemplatePose<>>(); }
