#pragma once

#include "brew/assert.hpp"
#include "brew/template_matching/pca_icp.hpp"
#include "brew/template_matching/point_cloud.hpp"
#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace brew::template_matching {

class TemplateLibrary {
public:
    struct Entry {
        std::shared_ptr<PointCloud> cloud;
        Eigen::Matrix3d pca_axes;
        Eigen::Vector3d pca_centroid;
    };

    TemplateLibrary() = default;

    TemplateLibrary(const TemplateLibrary&) = delete;
    TemplateLibrary& operator=(const TemplateLibrary&) = delete;
    TemplateLibrary(TemplateLibrary&&) = delete;
    TemplateLibrary& operator=(TemplateLibrary&&) = delete;

    int add(std::shared_ptr<PointCloud> cloud) {
        BREW_ASSERT(cloud && cloud->dim() == 3,
                    "TemplateLibrary only supports 3D templates");
        BREW_ASSERT(cloud->num_points() >= 3,
                    "TemplateLibrary requires at least 3 points for PCA");

        Entry entry;
        entry.pca_axes = PcaIcp::pca_axes(cloud->points());
        entry.pca_centroid = cloud->points().rowwise().mean();
        entry.cloud = std::move(cloud);

        std::unique_lock lock(mutex_);
        entries_.push_back(std::move(entry));
        return static_cast<int>(entries_.size()) - 1;
    }

    int add(PointCloud cloud) {
        return add(std::make_shared<PointCloud>(std::move(cloud)));
    }

    int add(std::shared_ptr<PointCloud> cloud,
            const Eigen::Matrix3d& pca_axes,
            const Eigen::Vector3d& pca_centroid) {
        BREW_ASSERT(cloud && cloud->dim() == 3,
                    "TemplateLibrary only supports 3D templates");
        BREW_ASSERT(cloud->num_points() >= 3,
                    "TemplateLibrary requires at least 3 points");

        Entry entry;
        entry.pca_axes = pca_axes;
        entry.pca_centroid = pca_centroid;
        entry.cloud = std::move(cloud);

        std::unique_lock lock(mutex_);
        entries_.push_back(std::move(entry));
        return static_cast<int>(entries_.size()) - 1;
    }

    int add(PointCloud cloud,
            const Eigen::Matrix3d& pca_axes,
            const Eigen::Vector3d& pca_centroid) {
        return add(std::make_shared<PointCloud>(std::move(cloud)),
                   pca_axes, pca_centroid);
    }

    [[nodiscard]] std::shared_ptr<PointCloud> get(int id) const {
        std::shared_lock lock(mutex_);
        BREW_ASSERT(id >= 0 && id < static_cast<int>(entries_.size()),
                    "TemplateLibrary::get id out of range");
        return entries_[id].cloud;
    }

    [[nodiscard]] Eigen::Matrix3d get_pca_axes(int id) const {
        std::shared_lock lock(mutex_);
        BREW_ASSERT(id >= 0 && id < static_cast<int>(entries_.size()),
                    "TemplateLibrary::get_pca_axes id out of range");
        return entries_[id].pca_axes;
    }

    [[nodiscard]] Eigen::Vector3d get_pca_centroid(int id) const {
        std::shared_lock lock(mutex_);
        BREW_ASSERT(id >= 0 && id < static_cast<int>(entries_.size()),
                    "TemplateLibrary::get_pca_centroid id out of range");
        return entries_[id].pca_centroid;
    }

    [[nodiscard]] Entry get_entry(int id) const {
        std::shared_lock lock(mutex_);
        BREW_ASSERT(id >= 0 && id < static_cast<int>(entries_.size()),
                    "TemplateLibrary::get_entry id out of range");
        return entries_[id];
    }

    [[nodiscard]] int size() const {
        std::shared_lock lock(mutex_);
        return static_cast<int>(entries_.size());
    }

private:
    mutable std::shared_mutex mutex_;
    std::vector<Entry> entries_;
};

}
