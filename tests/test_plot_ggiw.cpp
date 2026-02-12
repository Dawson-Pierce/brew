#include <gtest/gtest.h>
#include "brew/plot_utils/plot_ggiw.hpp"
#include "brew/plot_utils/math_utils.hpp"
#include <filesystem>

using namespace brew::plot_utils;
using namespace brew::distributions;

class PlotGGIWTest : public ::testing::Test {
protected:
    void SetUp() override {
        out_dir_ = "/workspace/brew/output";
        std::filesystem::create_directories(out_dir_);
    }

    std::filesystem::path out_dir_;
};

TEST_F(PlotGGIWTest, Plot2DNoThrow) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 20.0, 2.0, 2.0, 10.0;
    GGIW g(mean, cov, 100.0, 10.0, 10.0, V);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    EXPECT_NO_THROW(plot_ggiw_2d(ax, g, {0, 1}, {0.0f, 0.0f, 0.4470f, 0.7410f}));
}

TEST_F(PlotGGIWTest, Plot3DNoThrow) {
    Eigen::VectorXd mean(6);
    mean << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd V = Eigen::MatrixXd::Identity(3, 3) * 20.0;
    GGIW g(mean, cov, 100.0, 10.0, 10.0, V);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    EXPECT_NO_THROW(plot_ggiw_3d(ax, g, {0, 1, 2}, {0.0f, 0.0f, 0.4470f, 0.7410f}));
}

TEST_F(PlotGGIWTest, Save2D) {
    Eigen::VectorXd mean(4);
    mean << 0.0, 0.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V(2, 2);
    V << 20.0, 2.0, 2.0, 10.0;
    GGIW g(mean, cov, 100.0, 10.0, 10.0, V);

    auto output = (out_dir_ / "ggiw_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    opts.output_file = output;

    plot_ggiw(g, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotGGIWTest, Save3D) {
    Eigen::VectorXd mean(6);
    mean << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd V = Eigen::MatrixXd::Identity(3, 3) * 20.0;
    GGIW g(mean, cov, 100.0, 10.0, 10.0, V);

    auto output = (out_dir_ / "ggiw_3d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1, 2};
    opts.output_file = output;

    plot_ggiw(g, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotGGIWTest, MeanExtentCorrectness) {
    double v = 10.0;
    Eigen::MatrixXd V(2, 2);
    V << 28.0, 0.0,
         0.0, 14.0;
    int d = 2;
    Eigen::MatrixXd mean_ext = V / (v - d - 1.0);
    EXPECT_NEAR(mean_ext(0, 0), 4.0, 1e-10);
    EXPECT_NEAR(mean_ext(1, 1), 2.0, 1e-10);
}


