#include "brew/advanced/clustering/multithreshold_watershed.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>

namespace brew::clustering {

MultiThresholdWatershed::MultiThresholdWatershed(std::vector<double> lon, std::vector<double> lat,
                                                 std::vector<double> thresholds, int n_min,
                                                 int min_distance, int min_size)
    : lon_(std::move(lon)), lat_(std::move(lat)), thresholds_(std::move(thresholds)),
      n_min_(n_min), min_distance_(min_distance), min_size_(min_size) {}

std::vector<Eigen::MatrixXd> MultiThresholdWatershed::cluster(const Eigen::MatrixXd& Z) const {
    const int nlat = static_cast<int>(Z.rows());
    const int nlon = static_cast<int>(Z.cols());
    if (nlat == 0 || nlon == 0) return {};

    auto on = [&](int i, int j) {
        const double v = Z(i, j);
        return std::isfinite(v) && v > 0.0;
    };
    auto lin = [nlat](int i, int j) { return i + j * nlat; };
    const std::size_t N = static_cast<std::size_t>(nlat) * nlon;

    std::vector<double> thr = thresholds_;
    std::sort(thr.begin(), thr.end());
    if (thr.empty()) thr.push_back(0.0);

    // --- 1. Multi-threshold features: peak of each connected region with Z >= level ---
    std::vector<int> cand;
    std::vector<double> cand_level;
    std::vector<char> visited(N, 0);
    std::vector<int> stack;
    for (double t : thr) {
        std::fill(visited.begin(), visited.end(), 0);
        for (int j = 0; j < nlon; ++j) {
            for (int i = 0; i < nlat; ++i) {
                const int root = lin(i, j);
                if (visited[static_cast<std::size_t>(root)] || !on(i, j) || Z(i, j) < t) continue;
                stack.clear();
                stack.push_back(root);
                visited[static_cast<std::size_t>(root)] = 1;
                int peak = root;
                double peak_v = Z(i, j);
                int count = 0;
                while (!stack.empty()) {
                    const int c = stack.back();
                    stack.pop_back();
                    ++count;
                    const int ci = c % nlat, cj = c / nlat;
                    if (Z(ci, cj) > peak_v) { peak_v = Z(ci, cj); peak = c; }
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            if (di == 0 && dj == 0) continue;
                            const int ii = ci + di, jj = cj + dj;
                            if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                            const int q = lin(ii, jj);
                            const std::size_t qz = static_cast<std::size_t>(q);
                            if (!visited[qz] && on(ii, jj) && Z(ii, jj) >= t) {
                                visited[qz] = 1;
                                stack.push_back(q);
                            }
                        }
                }
                if (count >= n_min_) {
                    cand.push_back(peak);
                    cand_level.push_back(t);
                }
            }
        }
    }

    // --- 2. Keep features strongest-first, drop any within min_distance of a kept one ---
    std::vector<int> order(cand.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (cand_level[a] != cand_level[b]) return cand_level[a] > cand_level[b];
        return Z(cand[a] % nlat, cand[a] / nlat) > Z(cand[b] % nlat, cand[b] / nlat);
    });

    std::vector<int> label(N, -1);
    std::vector<char> blocked(N, 0);
    std::vector<int> seeds;
    int nseed = 0;
    const int r = std::max(min_distance_, 0);
    for (int oi : order) {
        const int p = cand[oi];
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

    // --- 3. Priority flood from features over the on-mask, high intensity to low ---
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
                if (on(ii, jj) && label[qz] == -1 && !queued[qz]) {
                    pq.push({Z(ii, jj), q});
                    queued[qz] = 1;
                }
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
                if (label[static_cast<std::size_t>(q)] >= 0) {
                    basin = label[static_cast<std::size_t>(q)];
                    break;
                }
            }
        }
        if (basin < 0) continue;
        label[static_cast<std::size_t>(cell)] = basin;
        push_nbrs(cell);
    }

    // --- 4. Emit feature basins (featureless regions dropped), drop small ---
    std::unordered_map<int, std::vector<int>> basins;
    for (int j = 0; j < nlon; ++j)
        for (int i = 0; i < nlat; ++i) {
            const int b = label[static_cast<std::size_t>(lin(i, j))];
            if (b >= 0) basins[b].push_back(lin(i, j));
        }

    const bool have_lon = static_cast<int>(lon_.size()) == nlon;
    const bool have_lat = static_cast<int>(lat_.size()) == nlat;

    std::vector<Eigen::MatrixXd> result;
    for (const auto& kv : basins) {
        const std::vector<int>& cells = kv.second;
        if (static_cast<int>(cells.size()) < min_size_) continue;
        Eigen::MatrixXd C(3, static_cast<Eigen::Index>(cells.size()));
        for (std::size_t w = 0; w < cells.size(); ++w) {
            const int p = cells[w];
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
