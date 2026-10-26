#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include <brew/ggiw/ggiw_model.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

void plot_ggiw_2d(matplot::axes_handle ax,
                  const brew::models::GGIW<>& g,
                  const std::vector<int>& plt_inds,
                  const Color& color,
                  double confidence = 0.99);

void plot_ggiw_3d(matplot::axes_handle ax,
                  const brew::models::GGIW<>& g,
                  const std::vector<int>& plt_inds,
                  const Color& color,
                  double confidence = 0.99,
                  float alpha = 0.3f,
                  float colormap_value = 0.0f);

void plot_ggiw(matplot::axes_handle ax,
               const brew::models::GGIW<>& g,
               const PlotOptions& opts);

void plot_ggiw(const brew::models::GGIW<>& g,
               const PlotOptions& opts);

}
