#pragma once

#include <Eigen/Dense>
#include <vector>

namespace brew::utils::clustering {

class ClusterBase {
public:
    virtual ~ClusterBase() = default;

    [[nodiscard]] virtual std::vector<Eigen::MatrixXd> cluster(
        const Eigen::MatrixXd& Z) const = 0;
};

}
