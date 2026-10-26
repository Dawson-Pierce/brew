#include <gtest/gtest.h>
#include "brew/gaussian/filters/ukf.hpp"
#include "brew/gaussian/filters/ekf.hpp"
#include "brew/dynamics/single_integrator.hpp"
#include "brew/dynamics/coordinated_turn.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace brew::filters;
using namespace brew::models;
using namespace brew::dynamics;

TEST(UKFTest, PredictStep) {
    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    UKF<> ukf;
    ukf.set_dynamics(dyn);

    Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
    Eigen::MatrixXd Q_in = 0.01 * Eigen::MatrixXd::Identity(G.cols(), G.cols());
    ukf.set_process_noise(G * Q_in * G.transpose());

    Eigen::VectorXd mean(4);
    mean << 0, 0, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);

    Gaussian prior(mean, cov);
    auto predicted = ukf.predict(1.0, prior);

    EXPECT_NEAR(predicted.mean()(0), 1.0, 1e-9);
    EXPECT_NEAR(predicted.mean()(1), 1.0, 1e-9);
    EXPECT_GT(predicted.covariance()(0, 0), cov(0, 0));
}

TEST(UKFTest, CorrectStep) {
    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    UKF<> ukf;
    ukf.set_dynamics(dyn);

    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    ukf.set_measurement_jacobian(H);
    ukf.set_measurement_noise(Eigen::MatrixXd::Identity(2, 2) * 0.1);

    Eigen::VectorXd mean(4);
    mean << 0, 0, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 10.0;
    Gaussian predicted(mean, cov);

    Eigen::VectorXd z(2);
    z << 0.5, 0.5;

    auto [corrected, likelihood] = ukf.correct(z, predicted);

    EXPECT_GT(corrected.mean()(0), 0.0);
    EXPECT_LT(corrected.mean()(0), 0.5);
    EXPECT_GT(likelihood, 0.0);
}

TEST(UKFTest, MatchesEKFOnLinearModel) {

    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
    Eigen::MatrixXd Q = G * (0.05 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose();
    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    Eigen::MatrixXd R = 0.2 * Eigen::MatrixXd::Identity(2, 2);

    EKF<> ekf;
    ekf.set_dynamics(dyn);
    ekf.set_process_noise(Q);
    ekf.set_measurement_jacobian(H);
    ekf.set_measurement_noise(R);

    UKF<> ukf;
    ukf.set_dynamics(dyn);
    ukf.set_process_noise(Q);
    ukf.set_measurement_jacobian(H);
    ukf.set_measurement_noise(R);

    Eigen::VectorXd mean(4);
    mean << 1, -2, 0.5, 0.3;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 2.0;
    cov(0, 2) = cov(2, 0) = 0.3;
    Gaussian prior(mean, cov);

    auto pe = ekf.predict(1.0, prior);
    auto pu = ukf.predict(1.0, prior);
    EXPECT_LT((pe.mean() - pu.mean()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((pe.covariance() - pu.covariance()).cwiseAbs().maxCoeff(), 1e-6);

    Eigen::VectorXd z(2);
    z << 1.4, -1.7;
    auto [ce, le] = ekf.correct(z, pe);
    auto [cu, lu] = ukf.correct(z, pu);
    EXPECT_LT((ce.mean() - cu.mean()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((ce.covariance() - cu.covariance()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_NEAR(le, lu, 1e-6 * std::max(1.0, le));
}

TEST(UKFTest, MatchesEKFAtLargeStateScale) {

    auto dyn = std::make_shared<SingleIntegrator<>>(2);
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, Eigen::VectorXd());
    Eigen::MatrixXd Q = G * (0.05 * Eigen::MatrixXd::Identity(G.cols(), G.cols())) * G.transpose();
    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    Eigen::MatrixXd R = 0.2 * Eigen::MatrixXd::Identity(2, 2);

    EKF<> ekf;
    ekf.set_dynamics(dyn); ekf.set_process_noise(Q);
    ekf.set_measurement_jacobian(H); ekf.set_measurement_noise(R);
    UKF<> ukf;
    ukf.set_dynamics(dyn); ukf.set_process_noise(Q);
    ukf.set_measurement_jacobian(H); ukf.set_measurement_noise(R);

    const double scale = 1e6;
    Eigen::VectorXd mean(4);
    mean << scale, -2 * scale, 0.5, 0.3;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 2.0;
    cov(0, 2) = cov(2, 0) = 0.3;
    Gaussian prior(mean, cov);

    auto pe = ekf.predict(1.0, prior);
    auto pu = ukf.predict(1.0, prior);
    EXPECT_LT((pe.mean() - pu.mean()).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT((pe.covariance() - pu.covariance()).cwiseAbs().maxCoeff(), 1e-5);

    Eigen::VectorXd z(2);
    z << pe.mean()(0) + 0.4, pe.mean()(1) - 0.7;
    auto [ce, le] = ekf.correct(z, pe);
    auto [cu, lu] = ukf.correct(z, pu);
    EXPECT_LT((ce.mean() - cu.mean()).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT((ce.covariance() - cu.covariance()).cwiseAbs().maxCoeff(), 1e-5);
}

namespace {

struct CTRun { double pos_rmse; double omega_est; };
template <typename Filt>
CTRun run_coordinated_turn(Filt& filt,
                           const std::vector<Eigen::VectorXd>& truth,
                           const std::vector<Eigen::VectorXd>& meas,
                           double dt, const Eigen::VectorXd& m0,
                           const Eigen::MatrixXd& P0) {
    Gaussian est(m0, P0);
    double sse = 0.0;
    int cnt = 0;
    const std::size_t half = meas.size() / 2;
    for (std::size_t k = 0; k < meas.size(); ++k) {
        est = filt.predict(dt, est);
        auto corrected = filt.correct(meas[k], est);
        est = corrected.distribution;
        if (k >= half) {
            const double ex = est.mean()(0) - truth[k](0);
            const double ey = est.mean()(1) - truth[k](1);
            sse += ex * ex + ey * ey;
            ++cnt;
        }
    }
    return { std::sqrt(sse / std::max(cnt, 1)), est.mean()(4) };
}
}

TEST(UKFTest, CoordinatedTurnUnknownRate) {

    using brew::dynamics::CoordinatedTurn;
    auto dyn = std::make_shared<CoordinatedTurn<>>();
    const double dt = 1.0;
    const double true_omega = 0.08;

    Eigen::VectorXd x(5);
    x << 0, 0, 10, 0, true_omega;
    std::mt19937 rng(42);
    std::normal_distribution<double> nz(0.0, std::sqrt(0.5));
    std::vector<Eigen::VectorXd> truth, meas;
    for (int k = 0; k < 40; ++k) {
        x = dyn->propagate_state(dt, x);
        truth.push_back(x);
        Eigen::VectorXd z(2);
        z << x(0) + nz(rng), x(1) + nz(rng);
        meas.push_back(z);
    }

    Eigen::MatrixXd H(2, 5);
    H.setZero();
    H(0, 0) = 1.0; H(1, 1) = 1.0;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(5, 5);
    Q(2, 2) = Q(3, 3) = 0.01; Q(4, 4) = 1e-3;
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd m0(5);
    m0 << 0, 0, 10, 0, 0.0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(5, 5);
    P0(4, 4) = 0.5;

    UKF<> ukf;
    ukf.set_dynamics(dyn); ukf.set_measurement_jacobian(H);
    ukf.set_process_noise(Q); ukf.set_measurement_noise(R);
    auto u = run_coordinated_turn(ukf, truth, meas, dt, m0, P0);

    EKF<> ekf;
    ekf.set_dynamics(dyn); ekf.set_measurement_jacobian(H);
    ekf.set_process_noise(Q); ekf.set_measurement_noise(R);
    auto e = run_coordinated_turn(ekf, truth, meas, dt, m0, P0);

    EXPECT_LT(u.pos_rmse, 3.0) << "UKF should track the turning target (rmse=" << u.pos_rmse << ")";
    EXPECT_NEAR(u.omega_est, true_omega, 0.05) << "UKF should converge the turn rate (got " << u.omega_est << ")";
    EXPECT_LT(std::abs(u.omega_est - true_omega), std::abs(e.omega_est - true_omega))
        << "UKF turn-rate error should beat the EKF (UKF=" << u.omega_est
        << ", EKF=" << e.omega_est << ", truth=" << true_omega << ")";
}
