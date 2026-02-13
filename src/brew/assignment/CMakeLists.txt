add_library(brew_assignment INTERFACE)
target_include_directories(brew_assignment INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_assignment INTERFACE Eigen3::Eigen)
