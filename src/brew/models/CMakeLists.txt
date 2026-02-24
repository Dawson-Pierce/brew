add_library(brew_models INTERFACE)

target_include_directories(brew_models INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(brew_models INTERFACE Eigen3::Eigen)
endif()
