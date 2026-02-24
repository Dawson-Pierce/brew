add_library(brew_clustering
    dbscan.cpp
    mahalanobis_dbscan.cpp
)
target_include_directories(brew_clustering PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(brew_clustering PUBLIC Eigen3::Eigen)
