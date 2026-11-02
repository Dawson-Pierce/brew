#pragma once

// Physical density interface plus optional downstream augmentation storage.

#include <Eigen/Dense>
#include <memory>
#include "brew/models/density_augmentation.hpp"

namespace brew::models {

class BaseSingleModel : public AugmentedState {
public:
    virtual ~BaseSingleModel() = default;

    [[nodiscard]] virtual std::unique_ptr<BaseSingleModel> clone() const = 0;

    [[nodiscard]] virtual bool is_extended() const { return false; }
};

}
