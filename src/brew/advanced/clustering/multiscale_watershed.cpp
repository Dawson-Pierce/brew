#include "brew/advanced/clustering/multiscale_watershed.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

namespace brew::clustering {

MultiScaleWatershed::MultiScaleWatershed(std::vector<double> lon, std::vector<double> lat,
                                         int region_dist, int min_seed_dist,
                                         int min_size, double min_core)
    : lon_(std::move(lon)), lat_(std::move(lat)),
      region_dist_(region_dist), min_seed_dist_(min_seed_dist),
      min_size_(min_size), min_core_(min_core) {}

std::vector<Eigen::MatrixXd> MultiScaleWatershed::cluster(const Eigen::MatrixXd& Z) const {
    const int nlat = static_cast<int>(Z.rows());
    const int nlon = static_cast<int>(Z.cols());
    if (nlat == 0 || nlon == 0) return {};

    auto on = [&](int i, int j) {
        const double v = Z(i, j);
        return std::isfinite(v) && v > 0.0;
    };
    auto lin = [nlat](int i, int j) { return i + j * nlat; };
    const std::size_t N = static_cast<std::size_t>(nlat) * nlon;

    // --- Local maxima within the mask (shared by both scales) ---
    std::vector<int> maxima;
    for (int j = 0; j < nlon; ++j) {
        for (int i = 0; i < nlat; ++i) {
            if (!on(i, j)) continue;
            const double v = Z(i, j);
            bool is_max = true;
            for (int dj = -1; dj <= 1 && is_max; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                    if (on(ii, jj) && Z(ii, jj) > v) { is_max = false; break; }
                }
            }
            if (is_max) maxima.push_back(lin(i, j));
        }
    }
    std::sort(maxima.begin(), maxima.end(), [&](int a, int b) {
        return Z(a % nlat, a / nlat) > Z(b % nlat, b / nlat);
    });

    // Seeds: maxima thinned to separation `rr`, highest kept first.
    auto seed_thin = [&](int rr, std::vector<int>& label, int& nseed) {
        std::vector<char> blocked(N, 0);
        std::vector<int> seeds;
        nseed = 0;
        const int r = std::max(rr, 0);
        for (int p : maxima) {
            if (blocked[static_cast<std::size_t>(p)]) continue;
            const int i = p % nlat, j = p / nlat;
            label[static_cast<std::size_t>(p)] = nseed++;
            seeds.push_back(p);
            const int j1 = std::max(0, j - r), j2 = std::min(nlon - 1, j + r);
            const int i1 = std::max(0, i - r), i2 = std::min(nlat - 1, i + r);
            for (int jj = j1; jj <= j2; ++jj)
                for (int ii = i1; ii <= i2; ++ii)
                    blocked[static_cast<std::size_t>(lin(ii, jj))] = 1;
        }
        return seeds;
    };

    // Priority flood from seeds, high intensity to low. When `region` is given a
    // basin may only grow into cells sharing the seed's region label.
    auto flood = [&](std::vector<int>& label, const std::vector<int>& seeds,
                     const std::vector<int>* region) {
        std::priority_queue<std::pair<double, int>> pq;
        std::vector<char> queued(N, 0);
        auto push_nbrs = [&](int cell) {
            const int i = cell % nlat, j = cell / nlat;
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                    const int q = lin(ii, jj);
                    const std::size_t qz = static_cast<std::size_t>(q);
                    if (!on(ii, jj) || label[qz] != -1 || queued[qz]) continue;
                    if (region && (*region)[qz] != (*region)[static_cast<std::size_t>(cell)])
                        continue;
                    pq.push({Z(ii, jj), q});
                    queued[qz] = 1;
                }
            }
        };
        for (int sc : seeds) push_nbrs(sc);
        while (!pq.empty()) {
            const int cell = pq.top().second;
            pq.pop();
            if (label[static_cast<std::size_t>(cell)] != -1) continue;
            const int i = cell % nlat, j = cell / nlat;
            int basin = -1;
            for (int dj = -1; dj <= 1 && basin == -1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                    const int q = lin(ii, jj);
                    if (label[static_cast<std::size_t>(q)] < 0) continue;
                    if (region && (*region)[static_cast<std::size_t>(q)] !=
                                      (*region)[static_cast<std::size_t>(cell)])
                        continue;
                    basin = label[static_cast<std::size_t>(q)];
                    break;
                }
            }
            if (basin < 0) continue;
            label[static_cast<std::size_t>(cell)] = basin;
            push_nbrs(cell);
        }
    };

    // Any on-cell left unlabeled gets its own basin (confined to `region`).
    auto fallback = [&](std::vector<int>& label, int& nlbl, const std::vector<int>* region) {
        for (int j = 0; j < nlon; ++j) {
            for (int i = 0; i < nlat; ++i) {
                if (!on(i, j) || label[static_cast<std::size_t>(lin(i, j))] != -1) continue;
                const int basin = nlbl++;
                const int root = lin(i, j);
                std::vector<int> stack{root};
                label[static_cast<std::size_t>(root)] = basin;
                while (!stack.empty()) {
                    const int c = stack.back();
                    stack.pop_back();
                    const int ci = c % nlat, cj = c / nlat;
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            if (di == 0 && dj == 0) continue;
                            const int ii = ci + di, jj = cj + dj;
                            if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                            const int q = lin(ii, jj);
                            const std::size_t qz = static_cast<std::size_t>(q);
                            if (!on(ii, jj) || label[qz] != -1) continue;
                            if (region && (*region)[qz] != (*region)[static_cast<std::size_t>(root)])
                                continue;
                            label[qz] = basin;
                            stack.push_back(q);
                        }
                }
            }
        }
    };

    // --- Coarse pass: carve the field into systems ---
    std::vector<int> coarse(N, -1);
    int ncoarse = 0;
    std::vector<int> cseeds = seed_thin(region_dist_, coarse, ncoarse);
    flood(coarse, cseeds, nullptr);
    fallback(coarse, ncoarse, nullptr);

    // --- Fine pass: split each system into cells, confined to its system ---
    std::vector<int> fine(N, -1);
    int nfine = 0;
    std::vector<int> fseeds = seed_thin(min_seed_dist_, fine, nfine);
    flood(fine, fseeds, &coarse);
    fallback(fine, nfine, &coarse);

    // --- Gather fine cells, drop small / weak-cored, emit [lon; lat; intensity] ---
    std::unordered_map<int, std::vector<int>> cells;
    for (int j = 0; j < nlon; ++j)
        for (int i = 0; i < nlat; ++i) {
            const int b = fine[static_cast<std::size_t>(lin(i, j))];
            if (b >= 0) cells[b].push_back(lin(i, j));
        }

    const bool have_lon = static_cast<int>(lon_.size()) == nlon;
    const bool have_lat = static_cast<int>(lat_.size()) == nlat;

    std::vector<Eigen::MatrixXd> result;
    for (const auto& kv : cells) {
        const std::vector<int>& pts = kv.second;
        if (static_cast<int>(pts.size()) < min_size_) continue;
        double peak = -std::numeric_limits<double>::infinity();
        for (int p : pts) peak = std::max(peak, Z(p % nlat, p / nlat));
        if (peak < min_core_) continue;
        Eigen::MatrixXd C(3, static_cast<Eigen::Index>(pts.size()));
        for (std::size_t w = 0; w < pts.size(); ++w) {
            const int p = pts[w];
            const int i = p % nlat, j = p / nlat;
            const auto col = static_cast<Eigen::Index>(w);
            C(0, col) = have_lon ? lon_[static_cast<std::size_t>(j)] : static_cast<double>(j);
            C(1, col) = have_lat ? lat_[static_cast<std::size_t>(i)] : static_cast<double>(i);
            C(2, col) = Z(i, j);
        }
        result.push_back(std::move(C));
    }
    return result;
}

} // namespace brew::clustering
