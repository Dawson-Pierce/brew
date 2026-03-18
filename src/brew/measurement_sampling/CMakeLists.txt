add_library(brew_measurement_sampling
    measurement_sampling.cpp
)
target_include_directories(brew_measurement_sampling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_measurement_sampling PUBLIC Eigen3::Eigen brew_template_matching)
