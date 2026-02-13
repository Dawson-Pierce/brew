#pragma once

#include <memory>

namespace brew::distributions {

/// Bernoulli component for multi-Bernoulli filters (MBM, PMBM).
/// Wraps a spatial distribution T with an existence probability.
/// T must provide: std::unique_ptr<T> clone_typed() const.
template <typename T>
class Bernoulli {
public:
    Bernoulli() = default;

    Bernoulli(double existence_prob, std::unique_ptr<T> distribution, int id = -1)
        : existence_prob_(existence_prob)
        , distribution_(std::move(distribution))
        , id_(id) {}

    [[nodiscard]] std::unique_ptr<Bernoulli<T>> clone() const {
        auto result = std::make_unique<Bernoulli<T>>();
        result->existence_prob_ = existence_prob_;
        if (distribution_) result->distribution_ = distribution_->clone_typed();
        result->id_ = id_;
        return result;
    }

    // ---- Accessors ----

    [[nodiscard]] double existence_probability() const { return existence_prob_; }
    void set_existence_probability(double r) { existence_prob_ = r; }

    [[nodiscard]] const T& distribution() const { return *distribution_; }
    [[nodiscard]] T& distribution() { return *distribution_; }
    void set_distribution(std::unique_ptr<T> dist) { distribution_ = std::move(dist); }

    [[nodiscard]] int id() const { return id_; }
    void set_id(int id) { id_ = id; }

    [[nodiscard]] bool has_distribution() const { return distribution_ != nullptr; }

private:
    double existence_prob_ = 0.0;
    std::unique_ptr<T> distribution_;
    int id_ = -1;
};

} // namespace brew::distributions
