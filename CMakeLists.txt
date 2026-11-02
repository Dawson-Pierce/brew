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
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/dependencies" CACHE PATH
    "Directory for fetched dependencies")

# Build-target switch. "desktop" builds the full library (plus sampling,
# plotting). "hardware" drops desktop-only modules and their deps
# (matplot++, STL samplers) for cross-compile friendliness.
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
    # Precondition checks use BREW_ASSERT instead of throw, so disable
    # exception machinery for smaller binaries and predictable control flow.
    if(MSVC)
        add_compile_definitions(_HAS_EXCEPTIONS=0)
    else()
        add_compile_options(-fno-exceptions)
    endif()
endif()

option(BREW_ENABLE_PLOTTING "Enable plotting utilities (matplot++, desktop only)" OFF)
option(BREW_BUILD_EXAMPLES  "Build examples (desktop only)" OFF)
option(BREW_BUILD_TESTS     "Build tests" OFF)

# Application builds use the installed Eigen headers and do not fetch source.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Headers and compiled sources are organized by responsibility.
file(GLOB BREW_FILTER_SOURCES CONFIGURE_DEPENDS src/brew/filters/*.cpp)
file(GLOB_RECURSE BREW_UTIL_SOURCES CONFIGURE_DEPENDS
    src/brew/utils/clustering/*.cpp src/brew/utils/registration/*.cpp)
file(GLOB BREW_DYNAMICS_SOURCES CONFIGURE_DEPENDS src/brew/dynamics/*.cpp)
add_library(brew STATIC ${BREW_FILTER_SOURCES} ${BREW_UTIL_SOURCES} ${BREW_DYNAMICS_SOURCES})
target_include_directories(brew PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(brew PUBLIC Eigen3::Eigen)

if(BREW_TARGET STREQUAL "desktop")
    target_sources(brew PRIVATE src/brew/utils/measurement_sampling/measurement_sampling.cpp)
    if(BREW_ENABLE_PLOTTING)
        FetchContent_Declare(matplotplusplus
            GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
            GIT_TAG v1.2.1)
        set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(matplotplusplus)
        file(GLOB BREW_PLOT_SOURCES CONFIGURE_DEPENDS src/brew/utils/plot_utils/*.cpp)
        target_sources(brew PRIVATE src/brew/utils/plot.cpp ${BREW_PLOT_SOURCES})
        target_link_libraries(brew PUBLIC matplot)
        target_compile_definitions(brew PUBLIC BREW_ENABLE_PLOTTING)
    endif()
endif()
if(BREW_TARGET STREQUAL "desktop" AND BREW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(BREW_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()
