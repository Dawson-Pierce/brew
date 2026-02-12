#include <gtest/gtest.h>
#include "brew/plot_utils/plot_mixture.hpp"
#include "brew/plot_utils/color_palette.hpp"
#include <filesystem>

using namespace brew::plot_utils;
using namespace brew::distributions;

class PlotMixtureTest : public ::testing::Test {
protected:
    void SetUp() override {
        out_dir_ = "/workspace/brew/output";
        std::filesystem::create_directories(out_dir_);
    }

    std::filesystem::path out_dir_;
};

TEST_F(PlotMixtureTest, GaussianMixture2DNoThrow) {
    Mixture<Gaussian> mix;
    Eigen::VectorXd m1(4); m1 << 0.0, 0.0, 1.0, 1.0;
    Eigen::VectorXd m2(4); m2 << 5.0, 5.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    mix.add_component(std::make_unique<Gaussian>(m1, cov), 0.6);
    mix.add_component(std::make_unique<Gaussian>(m2, cov * 2.0), 0.4);

    auto fig = matplot::figure(true);
    auto ax = fig->current_axes();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    EXPECT_NO_THROW(plot_mixture(ax, mix, opts));
}

TEST_F(PlotMixtureTest, SaveGaussianMixture) {
    Mixture<Gaussian> mix;
    Eigen::VectorXd m1(4); m1 << 0.0, 0.0, 1.0, 1.0;
    Eigen::VectorXd m2(4); m2 << 5.0, 5.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    mix.add_component(std::make_unique<Gaussian>(m1, cov), 0.6);
    mix.add_component(std::make_unique<Gaussian>(m2, cov * 2.0), 0.4);

    auto output = (out_dir_ / "gauss_mix_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    opts.output_file = output;

    plot_mixture(mix, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotMixtureTest, SaveGGIWMixture) {
    Mixture<GGIW> mix;
    Eigen::VectorXd m1(4); m1 << 0.0, 0.0, 1.0, 1.0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd V = Eigen::MatrixXd::Identity(2, 2) * 20.0;
    mix.add_component(std::make_unique<GGIW>(m1, cov, 100.0, 10.0, 10.0, V), 0.7);
    Eigen::VectorXd m2(4); m2 << 5.0, 3.0, 1.0, 1.0;
    mix.add_component(std::make_unique<GGIW>(m2, cov, 100.0, 10.0, 10.0, V), 0.3);

    auto output = (out_dir_ / "ggiw_mix_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    opts.output_file = output;

    plot_mixture(mix, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST_F(PlotMixtureTest, SaveTrajectoryMixture) {
    Mixture<TrajectoryGaussian> mix;
    Eigen::VectorXd m1(12);
    m1 << 0, 0, 1, 1,  1, 1, 1, 1,  2, 2, 1, 1;
    Eigen::MatrixXd cov1 = Eigen::MatrixXd::Identity(12, 12);
    mix.add_component(std::make_unique<TrajectoryGaussian>(0, 4, m1, cov1), 0.5);
    Eigen::VectorXd m2(12);
    m2 << 5, 5, -1, -1,  4, 4, -1, -1,  3, 3, -1, -1;
    mix.add_component(std::make_unique<TrajectoryGaussian>(0, 4, m2, cov1), 0.5);

    auto output = (out_dir_ / "traj_gauss_mix_2d.png").string();
    PlotOptions opts;
    opts.plt_inds = {0, 1};
    opts.output_file = output;

    plot_mixture(mix, opts);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_GT(std::filesystem::file_size(output), 0);
}

TEST(ColorPalette, LinesColorsMatch) {
    auto colors = lines_colors(7);
    ASSERT_EQ(colors.size(), 7);
    // ARGB format: {Alpha, Red, Green, Blue}
    EXPECT_NEAR(colors[0][0], 0.0f, 0.001f);     // alpha
    EXPECT_NEAR(colors[0][1], 0.0000f, 0.001f);   // R
    EXPECT_NEAR(colors[0][2], 0.4470f, 0.001f);   // G
    EXPECT_NEAR(colors[0][3], 0.7410f, 0.001f);   // B
    EXPECT_NEAR(colors[1][0], 0.0f, 0.001f);      // alpha
    EXPECT_NEAR(colors[1][1], 0.8500f, 0.001f);   // R
    EXPECT_NEAR(colors[1][2], 0.3250f, 0.001f);   // G
    EXPECT_NEAR(colors[1][3], 0.0980f, 0.001f);   // B
}

TEST(ColorPalette, CyclesCorrectly) {
    auto c7 = lines_color(0);
    auto c14 = lines_color(7);
    EXPECT_FLOAT_EQ(c7[0], c14[0]);
    EXPECT_FLOAT_EQ(c7[1], c14[1]);
    EXPECT_FLOAT_EQ(c7[2], c14[2]);
}


