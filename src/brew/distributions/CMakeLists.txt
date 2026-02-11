add_library(brew_distributions INTERFACE)

target_include_directories(brew_distributions INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(brew_distributions INTERFACE Eigen3::Eigen)
endif()
