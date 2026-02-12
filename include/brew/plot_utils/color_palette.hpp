#pragma once

#include "brew/plot_utils/plot_options.hpp"
#include <vector>

namespace brew::plot_utils {

/// Generate n colors matching MATLAB's default lines(n) palette.
/// Cycles through the 7-color MATLAB default palette.
std::vector<Color> lines_colors(int n);

/// Get a single color from the MATLAB lines palette (0-indexed, wraps).
Color lines_color(int index);

} // namespace brew::plot_utils

