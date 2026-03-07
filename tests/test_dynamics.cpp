#include <gtest/gtest.h>
#include "brew/dynamics/single_integrator.hpp"
#include "brew/dynamics/double_integrator.hpp"
#include "brew/dynamics/constant_turn_2d.hpp"

using namespace brew::dynamics;

TEST(SingleIntegrator2D, StateMatrix) {
    SingleIntegrator dyn(2);
    double dt = 0.1;
    auto F = dyn.get_state_mat(dt);

    EXPECT_EQ(F.rows(), 4);
    EXPECT_EQ(F.cols(), 4);
    EXPECT_DOUBLE_EQ(F(0, 2), dt);
    EXPECT_DOUBLE_EQ(F(1, 3), dt);
}

TEST(SingleIntegrator2D, Propagation) {
    SingleIntegrator dyn(2);
    Eigen::VectorXd state(4);
    state << 0, 0, 1, 1; // at origin, moving diagonally

    auto next = dyn.propagate_state(1.0, state);
    EXPECT_DOUBLE_EQ(next(0), 1.0); // x = 0 + 1*1
    EXPECT_DOUBLE_EQ(next(1), 1.0); // y = 0 + 1*1
    EXPECT_DOUBLE_EQ(next(2), 1.0); // vx unchanged
    EXPECT_DOUBLE_EQ(next(3), 1.0); // vy unchanged
}

TEST(SingleIntegrator1D, Propagation) {
    SingleIntegrator dyn(1);
    Eigen::VectorXd state(2);
    state << 5.0, 2.0;

    auto next = dyn.propagate_state(0.5, state);
    EXPECT_DOUBLE_EQ(next(0), 6.0); // 5 + 2*0.5
    EXPECT_DOUBLE_EQ(next(1), 2.0); // vx constant (no input)
}

TEST(DoubleIntegrator2D, StateNames) {
    DoubleIntegrator dyn(2);
    auto names = dyn.state_names();
    EXPECT_EQ(names.size(), 6u);
    EXPECT_EQ(names[0], "x");
    EXPECT_EQ(names[4], "ax");
}

TEST(ConstantTurn2D, StraightLine) {
    ConstantTurn2D dyn;
    Eigen::VectorXd state(5);
    state << 0, 0, 10, 0, 0; // v=10, theta=0, omega=0

    auto next = dyn.propagate_state(1.0, state);
    EXPECT_NEAR(next(0), 10.0, 1e-10); // x = v*cos(0)*dt
    EXPECT_NEAR(next(1), 0.0, 1e-10);  // y = v*sin(0)*dt
}
