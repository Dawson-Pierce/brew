#pragma once

#include "brew/desktop/plot_utils/plot_options.hpp"
#include <brew/gaussian/gaussian_model.hpp>
#include <matplot/matplot.h>

namespace brew::plot_utils {

void plot_gaussian(matplot::axes_handle ax,
                   const brew::models::Gaussian<>& g,
                   const std::vector<int>& plt_inds,
                   const Color& color,
                   double num_std = 2.0);

void plot_gaussian_2d(matplot::axes_handle ax,
                      const brew::models::Gaussian<>& g,
                      const std::vector<int>& plt_inds,
                      const Color& color,
                      double num_std = 2.0,
                      float alpha = 0.3f);

void plot_gaussian_3d(matplot::axes_handle ax,
                      const brew::models::Gaussian<>& g,
                      const std::vector<int>& plt_inds,
                      const Color& color,
                      double num_std = 2.0,
                      float alpha = 0.3f,
                      float colormap_value = 0.0f);

void plot_gaussian(matplot::axes_handle ax,
                   const brew::models::Gaussian<>& g,
                   const PlotOptions& opts);

void plot_gaussian(const brew::models::Gaussian<>& g,
                   const PlotOptions& opts);

}
