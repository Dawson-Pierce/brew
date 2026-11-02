// Compile-smoke + clone for the concrete gaussian RFS (MBMBase abstract -> only via inheritance asserts).

#include <gtest/gtest.h>
#include "brew/models/gaussian.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include "brew/multiObjectTrackers/mbm_base.hpp"
#include "brew/multiObjectTrackers/cphd.hpp"
#include "brew/multiObjectTrackers/glmb.hpp"
#include "brew/multiObjectTrackers/jglmb.hpp"
#include "brew/multiObjectTrackers/mbm.hpp"
#include "brew/multiObjectTrackers/pmbm.hpp"
#include "brew/models/merge/gaussian.hpp"
#include "brew/utils/fusion/gaussian_gci.hpp"
#include "brew/utils/fusion/arithmetic_average.hpp"
#include "brew/utils/fusion/prune.hpp"
#include "brew/utils/fusion/cap.hpp"
#include "brew/utils/clustering/dbscan.hpp"

using namespace brew;

template <typename Rfs>
static void expect_default_constructible_and_cloneable() {
    Rfs rfs;
    auto cloned = rfs.clone();
    EXPECT_NE(cloned, nullptr);
    static_assert(std::is_base_of_v<multiObjectTrackers::RFSBase, Rfs>,
                  "concrete gaussian RFS must derive from multiObjectTrackers::RFSBase");
}

TEST(GaussianConcreteRFS, AllInstantiateAndClone) {
    expect_default_constructible_and_cloneable<brew::multiObjectTrackers::PHD<brew::models::Gaussian<>, Eigen::Dynamic>>();
    expect_default_constructible_and_cloneable<brew::multiObjectTrackers::CPHD<brew::models::Gaussian<>, Eigen::Dynamic>>();
    expect_default_constructible_and_cloneable<brew::multiObjectTrackers::GLMB<brew::models::Gaussian<>, Eigen::Dynamic>>();
    expect_default_constructible_and_cloneable<brew::multiObjectTrackers::JGLMB<brew::models::Gaussian<>, Eigen::Dynamic>>();
    expect_default_constructible_and_cloneable<brew::multiObjectTrackers::MBM<brew::models::Gaussian<>, Eigen::Dynamic>>();
    expect_default_constructible_and_cloneable<brew::multiObjectTrackers::PMBM<brew::models::Gaussian<>, Eigen::Dynamic>>();
}

TEST(GaussianConcreteRFS, InheritanceChain) {
    static_assert(std::is_base_of_v<brew::multiObjectTrackers::GLMB<brew::models::Gaussian<>, Eigen::Dynamic>, brew::multiObjectTrackers::JGLMB<brew::models::Gaussian<>, Eigen::Dynamic>>,
                  "JGLMB must extend the gaussian GLMB");
    static_assert(std::is_base_of_v<brew::multiObjectTrackers::MBMBase<brew::models::Gaussian<>, Eigen::Dynamic>, brew::multiObjectTrackers::MBM<brew::models::Gaussian<>, Eigen::Dynamic>>,
                  "MBM must extend the gaussian MBMBase");
    static_assert(std::is_base_of_v<brew::multiObjectTrackers::MBMBase<brew::models::Gaussian<>, Eigen::Dynamic>, brew::multiObjectTrackers::PMBM<brew::models::Gaussian<>, Eigen::Dynamic>>,
                  "PMBM must extend the gaussian MBMBase");
    SUCCEED();
}
