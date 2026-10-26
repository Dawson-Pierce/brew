add_library(brew_dynamics STATIC
    linear_dynamics.cpp
    double_integrator_3d_euler.cpp
)
target_link_libraries(brew_dynamics PUBLIC brew_shared)
