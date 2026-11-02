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
#include "brew/models/ggiw.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/models/merge/ggiw.hpp"
#include "brew/utils/fusion/ggiw_gci.hpp"
#include "brew/dynamics/single_integrator.hpp"
#include <type_traits>

// A density that is not a BREW distribution subclass proves that the RFS layer
// depends on its interface, rather than an enumerated list of model families.
struct MarkedDensity {
    using Vector = Eigen::Vector2d;
    Vector state = Vector::Zero();
    int misses = 0;
    const Vector& mean() const { return state; }
    bool is_extended() const { return false; }
    std::unique_ptr<MarkedDensity> clone_typed() const {
        return std::make_unique<MarkedDensity>(*this);
    }
};

namespace brew::models {
template<> struct MixtureMerge<MarkedDensity> {
    template<int N>
    static void apply(Mixture<MarkedDensity,N>& mixture, double threshold, double) {
        if (threshold <= 0. || mixture.size() < 2) return;
        double weight = 0.;
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        for (std::size_t i=0; i<mixture.size(); ++i) {
            weight += mixture.weight(i);
            mean += mixture.weight(i)*mixture.component(i).mean();
        }
        auto density = mixture.component(0);
        density.state = mean/weight;
        mixture.clear();
        mixture.add_component(density, weight);
    }
};
}

class MarkedFilter : public brew::filters::Filter<MarkedDensity> {
public:
    double rate = 2.;
    std::unique_ptr<Filter> clone() const override { return std::make_unique<MarkedFilter>(*this); }
    MarkedDensity predict(double dt, const MarkedDensity& density) const override {
        auto result = density;
        result.state(0) += rate*dt;
        return result;
    }
    CorrectionResult correct(const MeasVector& measurement, const MarkedDensity& density) const override {
        auto result = density;
        result.state = (density.state+measurement)/2.;
        return {result, .25};
    }
    double gate(const MeasVector&, const MarkedDensity&) const override { return 0.; }
    std::vector<WeightedDensity> missed_detection(const MarkedDensity& density, double pd, double weight) const override {
        auto result = density;
        ++result.misses;
        return {{result, weight*(1.-pd)}};
    }
};

TEST(GenericRFS, FiltersMustMatchTheirDensity) {
    static_assert(!std::is_constructible_v<brew::multiObjectTrackers::PHD<brew::models::GGIW<>>, brew::filters::EKF<>>);
    SUCCEED();
}

TEST(GenericRFS, DynamicsFilterAndDensityConstructTheEstimatorDirectly) {
    brew::dynamics::SingleIntegrator<double,2> motion;
    brew::filters::GGIWEKF<double,4,2> filter(motion);
    filter.set_process_noise(Eigen::Matrix4d::Zero());
    brew::multiObjectTrackers::PHD<brew::models::GGIW<double,4,2>> phd(filter);
    Eigen::Vector4d mean; mean << 0.,0.,1.,2.;
    phd.intensity().add_component(brew::models::GGIW<double,4,2>(
        8.,2.,mean,Eigen::Matrix4d::Identity(),10.,4.*Eigen::Matrix2d::Identity()),1.);
    phd.predict(0, 2.);
    EXPECT_DOUBLE_EQ(phd.intensity().component(0).mean()(0), 2.);
    EXPECT_DOUBLE_EQ(phd.intensity().component(0).mean()(1), 4.);
}

TEST(GenericRFS, NewDensityUsesItsFilterForPredictionAndMissedUpdates) {
    MarkedFilter filter;
    brew::multiObjectTrackers::PHD<MarkedDensity> phd(filter);
    filter.rate = 100.; // PHD owns its configured filter clone.
    phd.intensity().add_component(MarkedDensity{}, .8);
    phd.set_prob_detection(.25);
    phd.set_prob_survive(.5);
    phd.predict(0, 3.);
    EXPECT_DOUBLE_EQ(phd.intensity().component(0).mean()(0), 6.);
    phd.correct(Eigen::MatrixXd(2,0));
    ASSERT_EQ(phd.intensity().size(), 1u);
    EXPECT_EQ(phd.intensity().component(0).misses, 1);
    EXPECT_DOUBLE_EQ(phd.expected_cardinality(), .8*.5*.75);
}

TEST(GenericRFS, AssociationWeightsAndMixtureMergingAreIndependentOfDensityMath) {
    MarkedFilter filter;
    brew::multiObjectTrackers::PHD<MarkedDensity> phd(filter);
    phd.intensity().add_component(MarkedDensity{}, .8);
    phd.set_prob_detection(.5);
    phd.set_clutter_rate(.1);
    phd.set_clutter_density(1.);
    Eigen::MatrixXd observation(2,1); observation << 4.,2.;
    phd.correct(observation);
    ASSERT_EQ(phd.intensity().size(), 2u);
    EXPECT_DOUBLE_EQ(phd.intensity().weight(0), .4);
    EXPECT_NEAR(phd.intensity().weight(1), .5, 1e-15);
    EXPECT_DOUBLE_EQ(phd.intensity().component(1).mean()(0), 2.);
    phd.cleanup();
    ASSERT_EQ(phd.intensity().size(), 1u);
    EXPECT_NEAR(phd.expected_cardinality(), .9, 1e-15);
    EXPECT_NEAR(phd.intensity().component(0).mean()(0), 1./.9, 1e-15);
}

TEST(GenericRFS, CloneContinuesTheSameRecursionAndRetainsHistoryLimit) {
    MarkedFilter filter;
    brew::multiObjectTrackers::PHD<MarkedDensity> original(filter);
    original.intensity().add_component(MarkedDensity{}, 1.);
    original.set_prob_detection(0.);
    original.set_max_history(2);
    original.step(Eigen::MatrixXd(2,0), 1.);
    auto clone = original.clone();
    auto& copied = dynamic_cast<brew::multiObjectTrackers::PHD<MarkedDensity>&>(*clone);
    for (int i=0; i<3; ++i) {
        original.step(Eigen::MatrixXd(2,0), 1.);
        copied.step(Eigen::MatrixXd(2,0), 1.);
    }
    EXPECT_EQ(copied.extracted_mixtures().size(), 2u);
    EXPECT_EQ(copied.intensity().component(0).misses, original.intensity().component(0).misses);
    EXPECT_EQ(copied.intensity().component(0).mean(), original.intensity().component(0).mean());
}

TEST(GenericRFS, FixedCapacityMixtureUsesTheSameSingleObjectRecursion) {
    MarkedFilter filter;
    brew::multiObjectTrackers::PHD<MarkedDensity,8> phd(filter);
    phd.intensity().add_component(MarkedDensity{}, 1.);
    phd.set_prob_detection(0.);
    phd.step(Eigen::MatrixXd(2,0), 2.);
    ASSERT_EQ(phd.intensity().size(), 1u);
    EXPECT_DOUBLE_EQ(phd.intensity().component(0).mean()(0), 4.);
    EXPECT_EQ(phd.intensity().component(0).misses, 1);
}

TEST(GenericRFS, AllRfsAlgorithmsAcceptAnUnlistedDensityAndItsFilter) {
    MarkedFilter filter;
    brew::multiObjectTrackers::CPHD<MarkedDensity> cphd(filter);
    brew::multiObjectTrackers::GLMB<MarkedDensity> glmb(filter);
    brew::multiObjectTrackers::JGLMB<MarkedDensity> jglmb(filter);
    brew::multiObjectTrackers::MBM<MarkedDensity> mbm(filter);
    brew::multiObjectTrackers::PMBM<MarkedDensity> pmbm(filter);
    EXPECT_NE(cphd.clone(), nullptr);
    EXPECT_NE(glmb.clone(), nullptr);
    EXPECT_NE(jglmb.clone(), nullptr);
    EXPECT_NE(mbm.clone(), nullptr);
    EXPECT_NE(pmbm.clone(), nullptr);
}
