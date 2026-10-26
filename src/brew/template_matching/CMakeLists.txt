add_library(brew_template_matching STATIC
    pca_icp.cpp
    point_to_plane_icp.cpp
    point_to_point_icp.cpp
)
target_link_libraries(brew_template_matching PUBLIC brew_shared)
