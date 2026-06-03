#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include <brew/ggiw_orientation/ggiw_orientation_model.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

void plot_ggiw_orientation_2d(matplot::axes_handle ax,
                              const brew::models::GGIWOrientation<>& g,
                              const std::vector<int>& plt_inds,
                              const Color& color,
                              double confidence = 0.99);

void plot_ggiw_orientation(matplot::axes_handle ax,
                           const brew::models::GGIWOrientation<>& g,
                           const PlotOptions& opts);

void plot_ggiw_orientation(const brew::models::GGIWOrientation<>& g,
                           const PlotOptions& opts);

}
