add_library(brew_metrics INTERFACE)
target_include_directories(brew_metrics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_metrics INTERFACE brew_assignment Eigen3::Eigen)
