#ifdef NDEBUG
#undef NDEBUG
#endif
#include "brew/dynamics/single_integrator.hpp"
#include "brew/filters/tm_ekf.hpp"
#include "brew/filters/ekf.hpp"
#include "brew/models/merge/gaussian.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace brew;
    auto motion = std::make_shared<dynamics::SingleIntegrator<>>(3);
    Eigen::MatrixXd Q = .01 * Eigen::MatrixXd::Identity(6,6);
    Eigen::MatrixXd R = .02 * Eigen::MatrixXd::Identity(6,6);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3,6);
    H.leftCols(3).setIdentity();
    Eigen::MatrixXd Qr = .03 * Eigen::Matrix3d::Identity();
    filters::TmEkf<> configured({.dynamics=motion, .process_noise=Q,
        .measurement_noise=R, .measurement_jacobian=H, .rotation_process_noise=Qr});
    filters::TmEkf<> legacy;
    legacy.set_dynamics(motion);
    legacy.set_process_noise(Q);
    legacy.set_measurement_noise(R);
    legacy.set_measurement_jacobian(H);
    legacy.set_rotation_process_noise(Qr);
    models::TemplatePose<> pose(Eigen::VectorXd::Ones(6), Eigen::MatrixXd::Identity(9,9), 0, {0,1,2});
    auto a = legacy.predict(1.,pose), b = configured.predict(1.,pose);
    assert(a.mean() == b.mean());
    assert(a.covariance() == b.covariance());
    assert(configured.measurement_noise() == legacy.measurement_noise());
    assert(configured.get_measurement_matrix(pose.mean()) == H);
    // Native callback configuration remains possible, not just matrix Jacobians.
    filters::TmEkf<>::Config callback;
    callback.measurement_jacobian = filters::TmEkf<>::Base::MeasurementJacobian{
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>([H](const auto&) {return H;})};
    filters::TmEkf<> with_callback(callback);
    assert(with_callback.get_measurement_matrix(pose.mean()) == H);

    using Density = models::Gaussian<double,4>;
    using Tracker = multiObjectTrackers::PHD<Density>;
    filters::EKF<double,4> primary;
    models::Mixture<Density> initial;
    initial.add_component(Density(Eigen::Vector4d::Zero(),Eigen::Matrix4d::Identity()),.8);
    Tracker phd(primary, {.prob_detection=0., .prob_survive=.7, .clutter_rate=0.,
        .clutter_density=.1, .max_history=1, .prune_threshold=0., .merge_threshold=0.,
        .max_components=10, .extract_threshold=0., .intensity=&initial});
    initial.component(0).mean().setOnes();
    assert(phd.prob_detection() == 0. && phd.prob_survive() == .7);
    assert(phd.clutter_rate() == 0. && phd.clutter_density() == .1 && phd.max_history() == 1);
    phd.cleanup();
    assert(phd.expected_cardinality() == .8);
    assert(phd.extract()->component(0).mean() == Eigen::Vector4d::Zero());
    bool rejected = false;
    try { Tracker bad(primary, {.prob_detection=1.1}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "Native configuration matches setters, preserves ownership, callbacks and validation\n";
}
