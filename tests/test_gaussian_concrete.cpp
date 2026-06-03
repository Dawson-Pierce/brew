// Compile-smoke + clone for the concrete gaussian RFS (MBMBase abstract -> only via inheritance asserts).

#include <gtest/gtest.h>
#include "brew/gaussian/gaussian.hpp"
#include "brew/gaussian/multi_target/mbm_base.hpp"

using namespace brew;

template <typename Rfs>
static void expect_default_constructible_and_cloneable() {
    Rfs rfs;
    auto cloned = rfs.clone();
    EXPECT_NE(cloned, nullptr);
    static_assert(std::is_base_of_v<multi_target::RFSBase, Rfs>,
                  "concrete gaussian RFS must derive from multi_target::RFSBase");
}

TEST(GaussianConcreteRFS, AllInstantiateAndClone) {
    expect_default_constructible_and_cloneable<gaussian::PHD<>>();
    expect_default_constructible_and_cloneable<gaussian::CPHD<>>();
    expect_default_constructible_and_cloneable<gaussian::GLMB<>>();
    expect_default_constructible_and_cloneable<gaussian::JGLMB<>>();
    expect_default_constructible_and_cloneable<gaussian::MBM<>>();
    expect_default_constructible_and_cloneable<gaussian::PMBM<>>();
}

TEST(GaussianConcreteRFS, InheritanceChain) {
    static_assert(std::is_base_of_v<gaussian::GLMB<>, gaussian::JGLMB<>>,
                  "JGLMB must extend the gaussian GLMB");
    static_assert(std::is_base_of_v<gaussian::MBMBase<>, gaussian::MBM<>>,
                  "MBM must extend the gaussian MBMBase");
    static_assert(std::is_base_of_v<gaussian::MBMBase<>, gaussian::PMBM<>>,
                  "PMBM must extend the gaussian MBMBase");
    SUCCEED();
}
