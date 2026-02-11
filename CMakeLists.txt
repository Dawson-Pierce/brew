cmake_minimum_required(VERSION 3.20)
project(BREW VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
include(FetchContent)

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

# Library modules (order matters: dependencies first)
add_subdirectory(src/brew/distributions)
add_subdirectory(src/brew/dynamics)
add_subdirectory(src/brew/filters)
add_subdirectory(src/brew/fusion)
add_subdirectory(src/brew/multi_target)
add_subdirectory(src/brew/clustering)

# Umbrella library that links all modules
add_library(brew INTERFACE)
target_link_libraries(brew INTERFACE
    brew_distributions
    brew_dynamics
    brew_filters
    brew_fusion
    brew_multi_target
    brew_clustering
)

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
