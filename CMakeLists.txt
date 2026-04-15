cmake_minimum_required(VERSION 3.20)
project(BREW VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MSVC: expose M_PI and other POSIX math constants
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# MinGW: allow large object files from heavy template instantiation (Eigen)
if(MINGW)
    add_compile_options(-Wa,-mbig-obj)
    # Link stdlib statically so launched exes don't need libgcc/libstdc++ DLLs
    # on PATH at runtime (fixes CTest gtest_discover_tests on bare shells).
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/dependencies" CACHE PATH
    "Directory for fetched dependencies")

# Build-target switch. "desktop" builds the full library (plus serialization,
# sampling, plotting). "hardware" drops desktop-only modules and their deps
# (yaml-cpp, matplot++, STL samplers) for cross-compile friendliness.
set(BREW_TARGET "desktop" CACHE STRING "Build target: desktop or hardware")
set_property(CACHE BREW_TARGET PROPERTY STRINGS desktop hardware)

if(NOT BREW_TARGET STREQUAL "desktop" AND NOT BREW_TARGET STREQUAL "hardware")
    message(FATAL_ERROR "BREW_TARGET must be 'desktop' or 'hardware' (got '${BREW_TARGET}')")
endif()
message(STATUS "BREW_TARGET: ${BREW_TARGET}")

# Hardware target forcibly disables desktop-only options.
if(BREW_TARGET STREQUAL "hardware")
    set(BREW_ENABLE_PLOTTING OFF CACHE BOOL "" FORCE)
    set(BREW_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(BREW_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
endif()

option(BREW_ENABLE_PLOTTING "Enable plotting utilities (matplot++, desktop only)" ON)
option(BREW_BUILD_EXAMPLES  "Build examples (desktop only)" ON)
option(BREW_BUILD_TESTS     "Build tests" ON)

# Eigen: system first, fallback to FetchContent
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally, fetching from GitLab...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# yaml-cpp: desktop only (used by serialization)
if(BREW_TARGET STREQUAL "desktop")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        0.8.0
        GIT_SHALLOW    TRUE
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# Library modules (order matters: dependencies first)
# core: hardware-portable hot path
add_subdirectory(src/brew/core/models)
add_subdirectory(src/brew/core/dynamics)
add_subdirectory(src/brew/core/template_matching)
add_subdirectory(src/brew/core/filters)
add_subdirectory(src/brew/core/fusion)
# advanced: multi-target / heavier algorithms
add_subdirectory(src/brew/advanced/assignment)
add_subdirectory(src/brew/advanced/multi_target)
add_subdirectory(src/brew/advanced/clustering)
add_subdirectory(src/brew/advanced/metrics)

# desktop: I/O, serialization, sampling, plotting (skipped on hardware)
if(BREW_TARGET STREQUAL "desktop")
    add_subdirectory(src/brew/desktop/serialization)
    add_subdirectory(src/brew/desktop/measurement_sampling)

    if(BREW_ENABLE_PLOTTING)
        FetchContent_Declare(
            matplotplusplus
            GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
            GIT_TAG v1.2.1
        )
        set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        # Suppress GCC 13+ warnings in matplot++ third-party code
        set(_prev_cxx_flags "${CMAKE_CXX_FLAGS}")
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation -Wno-ignored-attributes")
        endif()
        FetchContent_MakeAvailable(matplotplusplus)
        set(CMAKE_CXX_FLAGS "${_prev_cxx_flags}")

        add_subdirectory(src/brew/desktop/plot_utils)
    endif()
endif()

# Umbrella library
add_library(brew INTERFACE)
target_link_libraries(brew INTERFACE
    brew_models
    brew_dynamics
    brew_template_matching
    brew_filters
    brew_fusion
    brew_assignment
    brew_multi_target
    brew_clustering
    brew_metrics
)
if(BREW_TARGET STREQUAL "desktop")
    target_link_libraries(brew INTERFACE
        brew_serialization
        brew_measurement_sampling
    )
    if(BREW_ENABLE_PLOTTING)
        target_link_libraries(brew INTERFACE brew_plot_utils)
        target_compile_definitions(brew INTERFACE BREW_ENABLE_PLOTTING)
    endif()
endif()

# Examples (desktop only)
if(BREW_TARGET STREQUAL "desktop" AND BREW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests (require at least core + advanced; some need desktop modules)
if(BREW_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found locally, fetching from GitHub...")
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    enable_testing()
    add_subdirectory(tests)
endif()
