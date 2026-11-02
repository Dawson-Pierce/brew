#ifdef NDEBUG
#undef NDEBUG
#endif
#include "brew/dynamics/single_integrator.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/filters/ukf.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/models/categorical_intent.hpp"
#include <cassert>
#include <iostream>

using namespace brew;

template<class Filter, class Density>
void check_prediction(Filter filter, Density prior) {
    using Motion = dynamics::SingleIntegrator<double,
        Filter::DistStateDim == Eigen::Dynamic ? Eigen::Dynamic : Filter::DistStateDim / 2>;
    auto motion = std::make_shared<Motion>(2);
    motion->set_impulse_model(dynamics::ImpulseModel(models::ImpulsePrior(.2, .01),
        Eigen::Vector2d::UnitX(), {2,3}));
    auto cloned = motion->clone();
    motion->set_impulse_model(dynamics::ImpulseModel(models::ImpulsePrior(.4, .3),
        Eigen::Vector2d::UnitY(), {2,3}));
    assert(cloned->impulse_model()->prior().maximum == .2);
    filter.set_dynamics(std::shared_ptr<typename Filter::DynamicsType>(std::move(cloned)));
    filter.set_process_noise(Eigen::Matrix4d::Zero());
    models::CategoricalIntent intent;
    intent.objectives = {"a", "b"}; intent.probabilities = Eigen::Vector2d(.8,.2);
    prior.set_augmentation(intent);
    models::Mixture<Density> mix;
    mix.add_component(prior, .8);
    filter.predict_mixture(2., mix);
    assert(mix.size() == 2);
    assert(std::abs(mix.weight(0)-.792) < 1e-14);
    assert(std::abs(mix.weight(1)-.008) < 1e-14);
    Eigen::Vector4d delta(.2,0,.1,0), response(2,0,1,0);
    assert((mix.component(1).mean()-mix.component(0).mean()).isApprox(delta,1e-12));
    const Eigen::Matrix4d added = (.2*.2/12.)*response*response.transpose();
    assert((mix.component(1).covariance()-mix.component(0).covariance()).isApprox(added,1e-12));
    for (std::size_t i=0;i<2;++i)
        assert(mix.component(i).template augmentation<models::CategoricalIntent>().probabilities == intent.probabilities);
}

int main() {
    models::Gaussian<> gaussian(Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity());
    check_prediction(filters::EKF<>(), gaussian);
    check_prediction(filters::UKF<>(), gaussian);
    check_prediction(filters::EKF<double,4>(), models::Gaussian<double,4>(Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()));
    check_prediction(filters::GGIWEKF<>(), models::GGIW<>(8.,2.,Eigen::Vector4d::Zero(),Eigen::Matrix4d::Identity(),10.,Eigen::Matrix2d::Identity()));

    dynamics::ImpulseModel body(models::ImpulsePrior(.2,.01), Eigen::Vector3d::UnitX(), {3,4,5}, true);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6,6);
    F.topRightCorner(3,3)=2.*Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R; R << 0,-1,0, 1,0,0, 0,0,1;
    auto effect=body.prediction(F,R,.01*Eigen::Matrix3d::Identity());
    assert(std::abs(effect.mean_shift(1)-.2)<1e-14);
    assert(std::abs(effect.mean_shift(4)-.1)<1e-14);
    assert(std::abs(effect.attitude_jacobian(0,2)+.2)<1e-14);
    assert(std::abs(effect.covariance(0,0)-(.2*.2/12.)*4.*.01)<1e-14);

    filters::EKF<> filter;
    auto motion=std::make_shared<dynamics::SingleIntegrator<>>(3);
    motion->set_impulse_model(body);
    filter.set_dynamics(motion); filter.set_process_noise(Eigen::MatrixXd::Zero(6,6));
    models::Mixture<models::Gaussian<>> mix;
    mix.add_component(models::Gaussian<>(Eigen::VectorXd::Zero(6),Eigen::MatrixXd::Identity(6,6)),1.);
    bool rejected=false;
    try { filter.predict_mixture(1.,mix); } catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected && mix.size()==1);
    std::cout << "Reusable impulse dynamics: EKF, UKF, GGIW, fixed dimensions, moments, attitude and ownership passed\n";
}
