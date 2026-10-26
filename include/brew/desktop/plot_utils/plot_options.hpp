#pragma once

#include <array>
#include <string>
#include <vector>
#include <matplot/matplot.h>

namespace brew::plot_utils {

using Color = std::array<float, 4>;

struct PlotOptions {
    std::string output_file;
    std::vector<int> plt_inds;
    double num_std = 2.0;
    double confidence = 0.99;
    float alpha = 0.3f;
    Color color = {0.0f, 0.0f, 0.4470f, 0.7410f};
    float colormap_value = 0.0f;
    int width = 800;
    int height = 600;
};

void save_figure(matplot::figure_handle fig, const std::string& filename);

}
