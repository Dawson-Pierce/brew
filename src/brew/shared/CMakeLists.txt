# Shared base classes, generic containers, and fusion primitives.
# Header-only: exposes the whole include tree + Eigen to everything that links it.
add_library(brew_shared INTERFACE)
target_include_directories(brew_shared INTERFACE ${CMAKE_SOURCE_DIR}/include)
if(TARGET Eigen3::Eigen)
    target_link_libraries(brew_shared INTERFACE Eigen3::Eigen)
endif()
