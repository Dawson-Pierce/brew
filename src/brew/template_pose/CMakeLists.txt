add_library(brew_pkg_template_pose STATIC
    filters/tm_ekf.cpp
)
target_link_libraries(brew_pkg_template_pose PUBLIC brew_shared brew_dynamics brew_template_matching brew_clustering brew_assignment)
