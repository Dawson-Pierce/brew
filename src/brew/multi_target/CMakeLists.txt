add_library(brew_multi_target INTERFACE)
target_include_directories(brew_multi_target INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_multi_target INTERFACE brew_filters brew_fusion brew_assignment brew_clustering brew_distributions)
