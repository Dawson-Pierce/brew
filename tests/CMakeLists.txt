include(GoogleTest)
include(FetchContent)

set(EIGENRAND_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(EIGENRAND_BUILD_TEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    eigenrand
    GIT_REPOSITORY https://github.com/bab2min/EigenRand.git
    GIT_TAG v0.5.1
)
FetchContent_MakeAvailable(eigenrand)

if(NOT TARGET EigenRand::EigenRand)
    add_library(EigenRand::EigenRand INTERFACE IMPORTED)
    target_include_directories(EigenRand::EigenRand INTERFACE ${eigenrand_SOURCE_DIR})
endif()

add_executable(brew_tests
    test_gaussian.cpp
    test_dynamics.cpp
    test_ekf.cpp
    test_dbscan.cpp
    test_ggiw_ekf.cpp
    test_trajectory_ekf.cpp
    test_merge.cpp
    test_phd.cpp
    test_phd_tracking.cpp
    test_phd_tracking_extra.cpp
    test_phd_trajectory_tracking.cpp
    test_eigenrand.cpp
)

target_link_libraries(brew_tests PRIVATE
    brew
    GTest::gtest_main
    EigenRand::EigenRand
)

gtest_discover_tests(brew_tests)

if(BREW_ENABLE_PLOTTING)
    add_executable(test_math_utils test_math_utils.cpp)
    target_link_libraries(test_math_utils PRIVATE brew GTest::gtest_main)
    add_test(NAME test_math_utils COMMAND test_math_utils)
endif()
