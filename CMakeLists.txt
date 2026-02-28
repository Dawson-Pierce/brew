cmake_minimum_required(VERSION 3.20)
project(BREW VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

<<<<<<< Updated upstream
# MSVC: expose M_PI and other POSIX math constants
=======
# MSVC: M_PI and other POSIX math constants
>>>>>>> Stashed changes
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Dependencies
include(FetchContent)

# Options
option(BREW_ENABLE_PLOTTING "Enable plotting utilities (matplot++)" ON)

# Eigen: try system install first, fall back to FetchContent
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

# nlohmann/json for serialization
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# Library modules (order matters: dependencies first)
add_subdirectory(src/brew/models)
add_subdirectory(src/brew/dynamics)
add_subdirectory(src/brew/filters)
add_subdirectory(src/brew/fusion)
add_subdirectory(src/brew/assignment)
add_subdirectory(src/brew/multi_target)
add_subdirectory(src/brew/clustering)
add_subdirectory(src/brew/metrics)
add_subdirectory(src/brew/serialization)
if(BREW_ENABLE_PLOTTING)
    # --- Fetch matplot++ ---
    FetchContent_Declare(
        matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG v1.2.1
    )
    # Disable matplot++ examples and tests to speed up build
    set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    # Suppress GCC 13+ warnings in matplot++ third-party code before building it
    set(_prev_cxx_flags "${CMAKE_CXX_FLAGS}")
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation -Wno-ignored-attributes")
    endif()
    FetchContent_MakeAvailable(matplotplusplus)
    set(CMAKE_CXX_FLAGS "${_prev_cxx_flags}")

    add_subdirectory(src/brew/plot_utils)
endif()

# Umbrella library that links all modules
add_library(brew INTERFACE)
target_link_libraries(brew INTERFACE
    brew_models
    brew_dynamics
    brew_filters
    brew_fusion
    brew_assignment
    brew_multi_target
    brew_clustering
    brew_metrics
    brew_serialization
)
if(BREW_ENABLE_PLOTTING)
    target_link_libraries(brew INTERFACE brew_plot_utils)
    target_compile_definitions(brew INTERFACE BREW_ENABLE_PLOTTING)
endif()

# Testing
option(BREW_BUILD_TESTS "Build tests" ON)
if(BREW_BUILD_TESTS)
    # GTest: try system install first, fall back to FetchContent
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
