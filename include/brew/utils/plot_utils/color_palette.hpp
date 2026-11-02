#pragma once

#include "brew/utils/plot_utils/plot_options.hpp"
#include <vector>

namespace brew::utils::plot_utils {

std::vector<Color> lines_colors(int n);

Color lines_color(int index);

}
