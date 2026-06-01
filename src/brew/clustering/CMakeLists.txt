add_library(brew_clustering STATIC
    cc_watershed.cpp
    dbscan.cpp
    grid_cc.cpp
    mahalanobis_dbscan.cpp
    multiscale_watershed.cpp
    multithreshold_watershed.cpp
    percentile_watershed.cpp
    watershed.cpp
)
target_link_libraries(brew_clustering PUBLIC brew_shared)
