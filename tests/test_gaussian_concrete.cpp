#include <gtest/gtest.h>

// Compile-smoke test for the per-package CONCRETE gaussian RFS filters. Each is
// templated only on data type (Scalar) + size/shape (D, MaxComponents), never on
// the model — so this file exists mainly to force full template instantiation and
// vtable emission of every concrete gaussian::<RFS> through the package umbrella,
// catching de-templating regressions at compile time. clone() exercises the
// RFSBase override (value-copying the members) on a default-constructed filter.
#include "brew/gaussian/gaussian.hpp"
#include "brew/gaussian/multi_target/mbm_base.hpp"

using namespace brew;

// Every concrete gaussian RFS default-constructs and clones to a fresh RFSBase.
template <typename Rfs>
static void expect_default_constructible_and_cloneable() {
    Rfs rfs;                                   // forces full instantiation
    auto cloned = rfs.clone();                 // forces RFSBase::clone() override
    EXPECT_NE(cloned, nullptr);
    static_assert(std::is_base_of_v<multi_target::RFSBase, Rfs>,
                  "concrete gaussian RFS must derive from multi_target::RFSBase");
}

TEST(GaussianConcreteRFS, AllInstantiateAndClone) {
    // MBMBase is the abstract base (MBM/PMBM provide predict/correct/cleanup), so it
    // is exercised only through the inheritance static_asserts below, not instantiated.
    expect_default_constructible_and_cloneable<gaussian::PHD<>>();
    expect_default_constructible_and_cloneable<gaussian::CPHD<>>();
    expect_default_constructible_and_cloneable<gaussian::GLMB<>>();
    expect_default_constructible_and_cloneable<gaussian::JGLMB<>>();
    expect_default_constructible_and_cloneable<gaussian::MBM<>>();
    expect_default_constructible_and_cloneable<gaussian::PMBM<>>();
}

// Spot-check the inheritance relationships the migration relies on (JGLMB extends
// GLMB; MBM/PMBM extend MBMBase) so a refactor that accidentally re-parents one of
// them to the generic shared template fails here.
TEST(GaussianConcreteRFS, InheritanceChain) {
    static_assert(std::is_base_of_v<gaussian::GLMB<>, gaussian::JGLMB<>>,
                  "JGLMB must extend the gaussian GLMB");
    static_assert(std::is_base_of_v<gaussian::MBMBase<>, gaussian::MBM<>>,
                  "MBM must extend the gaussian MBMBase");
    static_assert(std::is_base_of_v<gaussian::MBMBase<>, gaussian::PMBM<>>,
                  "PMBM must extend the gaussian MBMBase");
    SUCCEED();
}
