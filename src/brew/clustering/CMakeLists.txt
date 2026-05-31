add_library(brew_clustering STATIC
    dbscan.cpp
    mahalanobis_dbscan.cpp
)
target_link_libraries(brew_clustering PUBLIC brew_shared)
