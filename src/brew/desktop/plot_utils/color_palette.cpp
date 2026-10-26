#include "brew/desktop/plot_utils/color_palette.hpp"

namespace brew::plot_utils {

namespace {

constexpr Color kMatlabPalette[7] = {
    {0.0f, 0.0000f, 0.4470f, 0.7410f},
    {0.0f, 0.8500f, 0.3250f, 0.0980f},
    {0.0f, 0.9290f, 0.6940f, 0.1250f},
    {0.0f, 0.4940f, 0.1840f, 0.5560f},
    {0.0f, 0.4660f, 0.6740f, 0.1880f},
    {0.0f, 0.3010f, 0.7450f, 0.9330f},
    {0.0f, 0.6350f, 0.0780f, 0.1840f},
};

}

std::vector<Color> lines_colors(int n) {
    std::vector<Color> colors(n);
    for (int i = 0; i < n; ++i) {
        colors[i] = kMatlabPalette[i % 7];
    }
    return colors;
}

Color lines_color(int index) {
    return kMatlabPalette[((index % 7) + 7) % 7];
}

}
