#ifdef NDEBUG
#undef NDEBUG
#endif
#include "brew/filters/tm_ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"
#include "brew/models/merge/template_pose.hpp"
#include <cassert>
#include <iostream>

using namespace brew;
using Pose = models::TemplatePose<>;

// A known registration isolates the evidence calculation from ICP local minima.
class ExactTranslation : public utils::registration::IcpBase {
public:
    utils::registration::IcpResult align(const utils::registration::PointCloud& source,
            const utils::registration::PointCloud& target, const Eigen::Matrix3d&,
            const Eigen::Vector3d&) const override {
        utils::registration::IcpResult result;
        result.translation = target.points().rowwise().mean() - source.points().rowwise().mean();
        result.log_likelihood = 0.;
        return result;
    }
    double log_likelihood(const utils::registration::PointCloud&, const utils::registration::PointCloud&,
            const Eigen::Matrix3d&, const Eigen::Vector3d&) const override { return 0.; }
    std::unique_ptr<IcpBase> clone() const override { return std::make_unique<ExactTranslation>(); }
};

// A controlled local minimum: a distant registration start fails, while a
// measurement-centered start finds the pose. Both priors must see that fit.
class LocalTranslation final : public ExactTranslation {
public:
    utils::registration::IcpResult align(const utils::registration::PointCloud& source,
            const utils::registration::PointCloud& target, const Eigen::Matrix3d& rotation,
            const Eigen::Vector3d& start) const override {
        auto result = ExactTranslation::align(source, target, rotation, start);
        if ((start-result.translation).norm() > 1.) result.translation = start;
        return result;
    }
    std::unique_ptr<IcpBase> clone() const override { return std::make_unique<LocalTranslation>(); }
};

int main() {
    Eigen::MatrixXd points(3,6);
    points << -2.,2.,0.,0.,0.,0., 0.,0.,-1.,1.,0.,0., 0.,0.,0.,0.,-.5,.5;
    auto library = std::make_shared<utils::registration::TemplateLibrary>();
    int id = library->add(utils::registration::PointCloud(points));
    auto motion = std::make_shared<dynamics::SingleIntegrator<>>(3);
    filters::TmEkf<> filter({.dynamics=motion,.template_library=library,.icp=std::make_shared<ExactTranslation>(),
        .process_noise=Eigen::MatrixXd::Zero(6,6),.measurement_noise=.01*Eigen::MatrixXd::Identity(6,6),
        .rotation_process_noise=Eigen::Matrix3d::Zero(),.pose_point_sigma_sq=.0625,
        .pose_angular_pitch=.01,.pose_return_cap=25});
    Eigen::VectorXd mean=Eigen::VectorXd::Zero(6); mean(0)=100.;
    Pose narrow(mean,.04*Eigen::MatrixXd::Identity(9,9),id,{0,1,2}); narrow.mark_pca_aligned();
    auto broad=narrow;
    broad.covariance().topLeftCorner(3,3)=144.*Eigen::Matrix3d::Identity();
    broad.covariance().bottomRightCorner(3,3)=.6*Eigen::Matrix3d::Identity();
    Eigen::MatrixXd measured=points.colwise()+Eigen::Vector3d(100.,0.,0.);
    const Eigen::VectorXd packed=Eigen::Map<Eigen::VectorXd>(measured.data(),measured.size());
    utils::registration::TmIcpRunner runner; runner.set_template_library(library);
    Eigen::MatrixXd R=.01*Eigen::MatrixXd::Identity(6,6);
    R.bottomRightCorner(3,3)=runner.pose_rotation_covariance(id,Eigen::Matrix3d::Identity(),100.,.0625,.01,25);
    Eigen::MatrixXd H=Eigen::MatrixXd::Zero(6,9);
    H.topLeftCorner(3,3).setIdentity(); H.bottomRightCorner(3,3).setIdentity();
    auto evidence=[&](const Pose& p) { return filter.correct_with_log(packed,p).second; };
    double expected=.5*std::log((H*broad.covariance()*H.transpose()+R).determinant()
                              /(H*narrow.covariance()*H.transpose()+R).determinant());
    assert(std::abs(evidence(narrow)-evidence(broad)-expected)<1e-9);
    auto displaced=narrow; displaced.mean()(0)+=2.;
    assert(std::abs(evidence(narrow)-evidence(displaced)-.5*4./.05)<1e-9);
    // The likelihood integral tends to the peak likelihood as prior uncertainty vanishes.
    auto certain=narrow; certain.covariance().setZero();
    double peak=runner.surface_log_likelihood(utils::registration::PointCloud(measured),id,
        Eigen::Matrix3d::Identity(),mean.head<3>(),.25,.05,4096.);
    filter.set_surface_sigma_sq(.25); filter.set_surface_outlier_probability(.05);
    filter.set_surface_outlier_volume(4096.);
    assert(std::abs(evidence(certain)-peak)<1e-9);

    motion->set_impulse_model(dynamics::ImpulseModel(models::ImpulsePrior(.2,.01),Eigen::Vector3d::UnitX(),{3,4,5},true));
    assert(std::isnan(filter.correct_with_log(packed,broad).first.distribution.firing().impulse_probability));
    std::vector<std::unique_ptr<Pose>> states;
    std::vector<double> priors,weights;
    for (int parent=0;parent<2;++parent) for (int branch=0;branch<2;++branch) {
        auto p=narrow.clone_typed();
        p->set_impulse_branch(branch,parent);
        p->set_impulse_log_likelihood(-10.*parent-branch);
        p->set_impulse_evidence(.1,.0001);
        priors.push_back(branch?.01:.99);
        weights.push_back(std::log(priors.back())+p->impulse_log_likelihood());
        states.push_back(std::move(p));
    }
    const auto original=weights;
    filter.adjust_log_weights(states,priors,weights);
    assert(weights==original); // A poor parent must not regain mass through sibling normalization.
    const double expected_burn=.01*std::exp(-1.)/(.99+.01*std::exp(-1.));
    for (const auto& p:states) assert(std::abs(p->firing().impulse_probability-expected_burn)<1e-12);

    models::Mixture<Pose> mixture;
    auto continuation=narrow; continuation.set_impulse_branch(0,0);
    continuation.set_firing(utils::firing_detail::firing_prior(.01,.2));
    mixture.add_component(continuation,.9); mixture.add_component(narrow,.1);
    mixture.reduce(0.,4.,20,.2);
    assert(mixture.size()==2); // A birth has no preceding interval to merge with this report.

    auto firing_probability=[&](double velocity_variance) {
        auto source=narrow;
        source.covariance().block(3,3,3,3)=velocity_variance*Eigen::Matrix3d::Identity();
        models::Mixture<Pose> prediction; prediction.add_component(source,1.);
        filter.predict_batch_dynamic(30.,prediction);
        Eigen::MatrixXd shifted=measured.colwise()+Eigen::Vector3d(3.,0.,0.);
        Eigen::VectorXd z=Eigen::Map<Eigen::VectorXd>(shifted.data(),shifted.size());
        std::vector<std::unique_ptr<Pose>> updated;
        std::vector<double> prior,log_weight;
        for(std::size_t i=0;i<prediction.size();++i) {
            auto [u,log_likelihood]=filter.correct_with_log(z,prediction.component(i));
            updated.push_back(u.distribution.clone_typed()); prior.push_back(prediction.weight(i));
            log_weight.push_back(std::log(prior.back())+log_likelihood);
        }
        filter.adjust_log_weights(updated,prior,log_weight);
        return updated.front()->firing().impulse_probability;
    };
    assert(firing_probability(.2*.2)<.05);
    assert(firing_probability(.001*.001)>.99);
    filter.set_icp(std::make_shared<LocalTranslation>());
    auto tracked = narrow;
    tracked.covariance().block(3,3,3,3) = 1e-6*Eigen::Matrix3d::Identity();
    models::Mixture<Pose> prediction; prediction.add_component(tracked,1.);
    filter.predict_batch_dynamic(30.,prediction);
    auto birth = broad; birth.mean()(0) = 103.;
    prediction.add_component(birth,.4);
    Eigen::MatrixXd shifted=measured.colwise()+Eigen::Vector3d(3.,0.,0.);
    Eigen::VectorXd z=Eigen::Map<Eigen::VectorXd>(shifted.data(),shifted.size());
    std::vector<const Pose*> gated_priors;
    for (std::size_t i=0;i<prediction.size();++i) gated_priors.push_back(&prediction.component(i));
    auto updates=filter.correct_batch_with_log(z,gated_priors);
    std::vector<std::unique_ptr<Pose>> updated;
    std::vector<double> prior,log_weight;
    for (std::size_t i=0;i<updates.size();++i) {
        updated.push_back(updates[i].first.distribution.clone_typed());
        prior.push_back(prediction.weight(i));
        log_weight.push_back(std::log(prior.back())+updates[i].second);
    }
    filter.adjust_log_weights(updated,prior,log_weight);
    const double continued=utils::firing_detail::log_add(log_weight[0],log_weight[1]);
    assert(std::exp(continued-utils::firing_detail::log_add(continued,log_weight[2]))>.999);
    assert(updated[1]->firing().impulse_probability>.99);
    assert(std::abs(updated[1]->mean()(3)-.1)<.002);
    assert(std::isnan(updated[2]->firing().impulse_probability));
    std::cout << "PASS: integrated pose evidence, parent competition, birth interval separation, velocity-aware impulse inference, and shared registration starts\n";
}
