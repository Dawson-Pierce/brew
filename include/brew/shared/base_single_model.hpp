#pragma once

// Notes: Pure data base class — clone() and virtual destructor only.

#include <Eigen/Dense>
#include <memory>

namespace brew::models {

class BaseSingleModel {
public:
    virtual ~BaseSingleModel() = default;

    [[nodiscard]] virtual std::unique_ptr<BaseSingleModel> clone() const = 0;

    [[nodiscard]] virtual bool is_extended() const { return false; }
};

}
