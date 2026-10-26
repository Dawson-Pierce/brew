#include <gtest/gtest.h>
#include "brew/dynamics/single_integrator.hpp"
#include "brew/dynamics/double_integrator.hpp"
#include "brew/dynamics/constant_turn_2d.hpp"
#include "brew/dynamics/coordinated_turn.hpp"
#include "brew/dynamics/double_integrator_3d_euler.hpp"
#include "brew/dynamics/clohessy_wiltshire.hpp"

using namespace brew::dynamics;

TEST(SingleIntegrator2D, StateMatrix) {
    SingleIntegrator<> dyn(2);
    double dt = 0.1;
    auto F = dyn.get_state_mat(dt);

    EXPECT_EQ(F.rows(), 4);
    EXPECT_EQ(F.cols(), 4);
    EXPECT_DOUBLE_EQ(F(0, 2), dt);
    EXPECT_DOUBLE_EQ(F(1, 3), dt);
}

TEST(SingleIntegrator2D, Propagation) {
    SingleIntegrator<> dyn(2);
    Eigen::VectorXd state(4);
    state << 0, 0, 1, 1;

    auto next = dyn.propagate_state(1.0, state);
    EXPECT_DOUBLE_EQ(next(0), 1.0);
    EXPECT_DOUBLE_EQ(next(1), 1.0);
    EXPECT_DOUBLE_EQ(next(2), 1.0);
    EXPECT_DOUBLE_EQ(next(3), 1.0);
}

TEST(SingleIntegrator1D, Propagation) {
    SingleIntegrator<> dyn(1);
    Eigen::VectorXd state(2);
    state << 5.0, 2.0;

    auto next = dyn.propagate_state(0.5, state);
    EXPECT_DOUBLE_EQ(next(0), 6.0);
    EXPECT_DOUBLE_EQ(next(1), 2.0);
}

TEST(DoubleIntegrator2D, StateNames) {
    DoubleIntegrator<> dyn(2);
    auto names = dyn.state_names();
    EXPECT_EQ(names.size(), 6u);
    EXPECT_EQ(names[0], "x");
    EXPECT_EQ(names[4], "ax");
}

TEST(ConstantTurn2DTest, StraightLine) {
    ConstantTurn2D<> dyn;
    Eigen::VectorXd state(5);
    state << 0, 0, 10, 0, 0;

    auto next = dyn.propagate_state(1.0, state);
    EXPECT_NEAR(next(0), 10.0, 1e-10);
    EXPECT_NEAR(next(1), 0.0, 1e-10);
}

TEST(ConstantTurn2DTest, PropagationDoesNotMultiplyByJacobian) {
    ConstantTurn2D<> dyn;
    Eigen::Matrix<double, 5, 1> state;
    state << 2.0, -1.0, 4.0, 0.7, 0.2;
    const double dt = 0.3;
    const auto next = dyn.propagate_state(dt, state);
    EXPECT_NEAR(next(0), state(0) + state(2) * std::cos(state(3)) * dt, 1e-12);
    EXPECT_NEAR(next(1), state(1) + state(2) * std::sin(state(3)) * dt, 1e-12);
    EXPECT_NEAR(next(3), state(3) + state(4) * dt, 1e-12);
}

TEST(CoordinatedTurnTest, JacobianMatchesFiniteDifference) {
    CoordinatedTurn<> dyn;
    Eigen::VectorXd state(5);
    state << 2.0, -1.0, 4.0, 1.5, 0.2;
    const double dt = 0.3;
    const double eps = 1e-6;
    const Eigen::MatrixXd F = dyn.get_state_mat(dt, state);
    Eigen::MatrixXd numerical(5, 5);
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd plus = state;
        Eigen::VectorXd minus = state;
        plus(i) += eps;
        minus(i) -= eps;
        numerical.col(i) = (dyn.propagate_state(dt, plus) - dyn.propagate_state(dt, minus))
            / (2.0 * eps);
    }
    EXPECT_TRUE(F.isApprox(numerical, 1e-7));
}

TEST(DoubleIntegrator3DEulerTest, JacobianMatchesFiniteDifference) {
    DoubleIntegrator3DEuler<> dyn;
    Eigen::Matrix<double, 18, 1> state;
    state << 1.0, 2.0, 3.0, 0.2, -0.1, 0.3, 0.01, 0.02, -0.03,
             0.2, -0.3, 0.4, 0.1, -0.2, 0.15, 0.03, -0.01, 0.02;
    const double dt = 0.2;
    const double eps = 1e-6;
    const auto F = dyn.get_state_mat(dt, state);
    Eigen::Matrix<double, 18, 18> numerical;
    for (int i = 0; i < 18; ++i) {
        auto plus = state;
        auto minus = state;
        plus(i) += eps;
        minus(i) -= eps;
        numerical.col(i) = (dyn.propagate_state(dt, plus) - dyn.propagate_state(dt, minus))
            / (2.0 * eps);
    }
    EXPECT_TRUE(F.isApprox(numerical, 1e-7));
}

TEST(ClohessyWiltshire, IdentityAtZeroTime) {
    ClohessyWiltshire<> dyn(0.0011);
    EXPECT_TRUE(dyn.get_state_mat(0.0).isApprox(Eigen::MatrixXd::Identity(6, 6)));
    EXPECT_TRUE(dyn.get_input_mat(0.0).isZero());
}

TEST(ClohessyWiltshire, CrossTrackMotionIsPeriodic) {
    const double n = 0.0011;
    ClohessyWiltshire<> dyn(n);
    Eigen::Matrix<double, 6, 1> state;
    state << 0.0, 0.0, 100.0, 0.0, 0.0, 0.03;
    const auto next = dyn.propagate_state(2.0 * M_PI / n, state);
    EXPECT_TRUE(next.isApprox(state, 1e-10));
}
