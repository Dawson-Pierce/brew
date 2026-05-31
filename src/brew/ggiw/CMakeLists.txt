add_library(brew_pkg_ggiw STATIC
    filters/ggiw_ekf.cpp
)
target_link_libraries(brew_pkg_ggiw PUBLIC brew_shared brew_dynamics brew_template_matching brew_clustering brew_assignment)
