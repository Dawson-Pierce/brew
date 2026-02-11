add_library(brew_fusion
    merge.cpp
)
target_include_directories(brew_fusion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_fusion PUBLIC brew_distributions Eigen3::Eigen)
