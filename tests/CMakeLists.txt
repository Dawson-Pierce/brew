include(GoogleTest)

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
)

target_link_libraries(brew_tests PRIVATE
    brew
    GTest::gtest_main
)

gtest_discover_tests(brew_tests)
