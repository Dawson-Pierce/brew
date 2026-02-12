#include <gtest/gtest.h>
#include "brew/plot_utils/plot_trajectory_gaussian.hpp"
#include "brew/plot_utils/plot_trajectory_ggiw.hpp"
#include <filesystem>

using namespace brew::plot_utils;
using namespace brew::distributions;

class PlotTrajectoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        out_dir_ = "/workspace/brew/output";
        std::filesystem::create_directories(out_dir_);
    }

    std::filesystem::path out_dir_;
};

TEST_F(PlotTrajectoryTest, TrajectoryGaussian1DNoThrow) {
    Eigen::VectorXd mean(20);
    for (int t = 0; t < 5; ++t) {
        mean(t * 4 + 0) = t * 1.0;
        mean(t * 4 + 1) = t * 0.5;
        mean(t * 4 + 2) = 1.0;
        mean(t * 4 + 3) = 0.5;
    }
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(20, 20);
    TrajectoryGaussian tg(0, 4, mean, cov);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    EXPECT_NO_THROW(plot_trajectory_gaussian_1d(ax, tg, {0}, {0.0f, 0.0f, 0.4470f, 0.7410f}));
}

TEST_F(PlotTrajectoryTest, SaveTrajectoryGaussian1D) {
    Eigen::VectorXd mean(20);
    for (int t = 0; t < 5; ++t) {
        mean(t * 4 + 0) = t * 1.0;
        mean(t * 4 + 1) = t * 0.5;
        mean(t * 4 + 2) = 1.0;
        mean(t * 4 + 3) = 0.5;
    }
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(20, 20);
    TrajectoryGaussian tg(0, 4, mean, cov);

    auto output = (out_dir_ / "traj_gauss_1d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0};
    opts.output_file = output;

    plot_trajectory_gaussian(tg, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotTrajectoryTest, SaveTrajectoryGaussian2D) {
    Eigen::VectorXd mean(20);
    for (int t = 0; t < 5; ++t) {
        mean(t * 4 + 0) = t * 1.0;
        mean(t * 4 + 1) = t * 0.5;
        mean(t * 4 + 2) = 1.0;
        mean(t * 4 + 3) = 0.5;
    }
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(20, 20);
    TrajectoryGaussian tg(0, 4, mean, cov);

    auto output = (out_dir_ / "traj_gauss_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    opts.output_file = output;

    plot_trajectory_gaussian(tg, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotTrajectoryTest, SaveTrajectoryGaussian3D) {
    Eigen::VectorXd mean(30);
    for (int t = 0; t < 5; ++t) {
        mean(t * 6 + 0) = t * 1.0;
        mean(t * 6 + 1) = t * 0.5;
        mean(t * 6 + 2) = t * 0.3;
        mean(t * 6 + 3) = 1.0;
        mean(t * 6 + 4) = 0.5;
        mean(t * 6 + 5) = 0.3;
    }
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(30, 30);
    TrajectoryGaussian tg(0, 6, mean, cov);

    auto output = (out_dir_ / "traj_gauss_3d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1, 2};
    opts.output_file = output;

    plot_trajectory_gaussian(tg, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotTrajectoryTest, SaveTrajectoryGGIW2D) {
    Eigen::VectorXd mean(12);
    for (int t = 0; t < 3; ++t) {
        mean(t * 4 + 0) = t * 2.0;
        mean(t * 4 + 1) = t * 1.0;
        mean(t * 4 + 2) = 2.0;
        mean(t * 4 + 3) = 1.0;
    }
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12);
    Eigen::MatrixXd V = Eigen::MatrixXd::Identity(2, 2) * 20.0;
    TrajectoryGGIW tg(0, 4, mean, cov, 100.0, 10.0, 10.0, V);

    auto output = (out_dir_ / "traj_ggiw_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    opts.output_file = output;

    plot_trajectory_ggiw(tg, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotTrajectoryTest, SaveTrajectoryGGIW3D) {
    Eigen::VectorXd mean(18);
    for (int t = 0; t < 3; ++t) {
        mean(t * 6 + 0) = t * 2.0;
        mean(t * 6 + 1) = t * 1.0;
        mean(t * 6 + 2) = t * 0.5;
        mean(t * 6 + 3) = 2.0;
        mean(t * 6 + 4) = 1.0;
        mean(t * 6 + 5) = 0.5;
    }
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(18, 18);
    Eigen::MatrixXd V = Eigen::MatrixXd::Identity(3, 3) * 20.0;
    TrajectoryGGIW tg(0, 6, mean, cov, 100.0, 10.0, 10.0, V);

    auto output = (out_dir_ / "traj_ggiw_3d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1, 2};
    opts.output_file = output;

    plot_trajectory_ggiw(tg, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}


