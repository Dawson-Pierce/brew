#include <gtest/gtest.h>
#include "brew/plot_utils/plot_gaussian.hpp"
#include <filesystem>

using namespace brew::plot_utils;
using namespace brew::distributions;

class PlotGaussianTest : public ::testing::Test {
protected:
    void SetUp() override {
        out_dir_ = "/workspace/brew/output";
        std::filesystem::create_directories(out_dir_);
    }

    std::filesystem::path out_dir_;
};

TEST_F(PlotGaussianTest, Plot1DNoThrow) {
    Eigen::VectorXd mean(2);
    mean << 5.0, 3.0;
    Eigen::MatrixXd cov(2, 2);
    cov << 2.0, 0.5,
           0.5, 1.0;
    Gaussian g(mean, cov);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    EXPECT_NO_THROW(plot_gaussian(ax, g, {0}, {0.0f, 0.0f, 0.4470f, 0.7410f}));
}

TEST_F(PlotGaussianTest, Plot2DNoThrow) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    cov(0, 0) = 4.0;
    cov(2, 2) = 9.0;
    Gaussian g(mean, cov);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    EXPECT_NO_THROW(plot_gaussian_2d(ax, g, {0, 2}, {0.0f, 0.0f, 0.4470f, 0.7410f}));
}

TEST_F(PlotGaussianTest, Plot3DNoThrow) {
    Eigen::VectorXd mean(3);
    mean << 1.0, 2.0, 3.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(3, 3) * 2.0;
    Gaussian g(mean, cov);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    EXPECT_NO_THROW(plot_gaussian_3d(ax, g, {0, 1, 2}, {0.0f, 0.0f, 0.4470f, 0.7410f}));
}

TEST_F(PlotGaussianTest, Save1D) {
    Eigen::VectorXd mean(2);
    mean << 5.0, 3.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);
    Gaussian g(mean, cov);

    auto output = (out_dir_ / "gaussian_1d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0};
    opts.output_file = output;

    plot_gaussian(g, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotGaussianTest, Save2D) {
    Eigen::VectorXd mean(4);
    mean << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Gaussian g(mean, cov);

    auto output = (out_dir_ / "gaussian_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 2};
    opts.output_file = output;

    plot_gaussian(g, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotGaussianTest, Save3D) {
    Eigen::VectorXd mean(3);
    mean << 1.0, 2.0, 3.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(3, 3) * 2.0;
    Gaussian g(mean, cov);

    auto output = (out_dir_ / "gaussian_3d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1, 2};
    opts.output_file = output;

    plot_gaussian(g, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotGaussianTest, InvalidIndsThrows) {
    Eigen::VectorXd mean(2);
    mean << 5.0, 3.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);
    Gaussian g(mean, cov);

    PlotOptions opts;
    opts.plt_inds = {0, 1, 2, 3};
    EXPECT_THROW(plot_gaussian(g, opts), std::invalid_argument);
}


