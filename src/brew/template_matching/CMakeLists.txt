add_library(brew_template_matching
    point_to_point_icp.cpp
    point_to_plane_icp.cpp
    point_cloud_io.cpp
    pca_icp.cpp
)
target_include_directories(brew_template_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_template_matching PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)
