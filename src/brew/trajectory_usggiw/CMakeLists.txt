add_library(brew_pkg_trajectory_usggiw STATIC
    filters/trajectory_usggiw_ekf.cpp
)
target_link_libraries(brew_pkg_trajectory_usggiw PUBLIC brew_shared brew_dynamics brew_clustering)

