add_library(brew_serialization INTERFACE)
target_include_directories(brew_serialization INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_serialization INTERFACE
    brew_models
    brew_multi_target
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)
