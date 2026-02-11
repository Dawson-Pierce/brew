add_library(brew_dynamics
    linear_dynamics.cpp
    double_integrator_3d_euler.cpp
)

target_include_directories(brew_dynamics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(brew_dynamics PUBLIC Eigen3::Eigen)
endif()
