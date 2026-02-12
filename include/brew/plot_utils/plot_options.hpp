#pragma once

#include <array>
#include <string>
#include <vector>
#include <matplot/matplot.h>

namespace brew::plot_utils {

/// ARGB color (each component in [0, 1]).  matplot++ uses Alpha/Red/Green/Blue
/// where Alpha=0 is opaque and Alpha=1 is fully transparent.
using Color = std::array<float, 4>;

/// Options controlling plot appearance and output.
struct PlotOptions {
    std::string output_file;          ///< If non-empty, save figure to this path
    std::vector<int> plt_inds;        ///< 0-based state indices to plot
    double num_std = 2.0;             ///< Number of std deviations for Gaussian ellipses
    double confidence = 0.99;         ///< Confidence level for GGIW extent ellipses
    float alpha = 0.3f;               ///< Fill transparency
    Color color = {0.0f, 0.0f, 0.4470f, 0.7410f};   ///< Default MATLAB blue (ARGB)
    float colormap_value = 0.0f;      ///< CData value for 3D surface coloring (used by mixtures)
    int width = 800;                  ///< Figure width in pixels
    int height = 600;                 ///< Figure height in pixels
};

/// Save a figure to file by explicitly setting gnuplot terminal and output.
/// Determines the terminal type from the file extension (png -> pngcairo, svg -> svg, etc.).
void save_figure(matplot::figure_handle fig, const std::string& filename);

} // namespace brew::plot_utils

