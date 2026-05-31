add_library(brew_pkg_gaussian STATIC
    filters/ekf.cpp
)
target_link_libraries(brew_pkg_gaussian PUBLIC brew_shared brew_dynamics brew_template_matching brew_clustering brew_assignment)
