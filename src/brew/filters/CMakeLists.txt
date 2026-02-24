add_library(brew_filters
    ekf.cpp
    ggiw_ekf.cpp
    trajectory_gaussian_ekf.cpp
    trajectory_ggiw_ekf.cpp
)
target_include_directories(brew_filters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_filters PUBLIC brew_models brew_dynamics Eigen3::Eigen)
