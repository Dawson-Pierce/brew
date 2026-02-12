add_library(brew_plot_utils
    color_palette.cpp
    math_utils.cpp
    plot_options.cpp
    plot_gaussian.cpp
    plot_ggiw.cpp
    plot_trajectory_gaussian.cpp
    plot_trajectory_ggiw.cpp
    plot_mixture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../plot.cpp
)

target_include_directories(brew_plot_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

target_link_libraries(brew_plot_utils PUBLIC
    brew_distributions
    matplot
)
