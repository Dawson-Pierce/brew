#include <gtest/gtest.h>
#include "brew/clustering/dbscan.hpp"

using namespace brew::clustering;

TEST(DBSCAN, TwoClusters) {
    // Create two well-separated clusters
    Eigen::MatrixXd Z(2, 6);
    Z << 0.0, 0.1, 0.2,  5.0, 5.1, 5.2,
         0.0, 0.1, 0.2,  5.0, 5.1, 5.2;

    DBSCAN db(0.5, 2);
    auto clusters = db.cluster(Z);

    EXPECT_EQ(clusters.size(), 2u);
}

TEST(DBSCAN, Noise) {
    Eigen::MatrixXd Z(2, 4);
    Z << 0.0, 0.1, 10.0, 20.0,
         0.0, 0.1, 10.0, 20.0;

    DBSCAN db(0.5, 2);
    auto noise = db.get_unclustered(Z);

    // Points at (10,10) and (20,20) are isolated noise
    EXPECT_EQ(noise.size(), 2u);
}

TEST(DBSCAN, Empty) {
    Eigen::MatrixXd Z(2, 0);
    DBSCAN db(1.0, 3);
    auto clusters = db.cluster(Z);
    EXPECT_TRUE(clusters.empty());
}
