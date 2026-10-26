#include "brew/clustering/percentile_watershed.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brew::clustering {

PercentileWatershed::PercentileWatershed(std::vector<double> lon, std::vector<double> lat,
                                         double percentile, int n_min, int min_distance,
                                         int min_size, double min_core_abs)
    : lon_(std::move(lon)), lat_(std::move(lat)),
      percentile_(percentile), n_min_(n_min), min_distance_(min_distance),
      min_size_(min_size), min_core_abs_(min_core_abs) {}

std::vector<Eigen::MatrixXd> PercentileWatershed::cluster(const Eigen::MatrixXd& Z) const {
    const int nlat = static_cast<int>(Z.rows());
    const int nlon = static_cast<int>(Z.cols());
    if (nlat == 0 || nlon == 0) return {};

    auto on = [&](int i, int j) {
        const double v = Z(i, j);
        return std::isfinite(v) && v > 0.0;
    };
    auto lin = [nlat](int i, int j) { return i + j * nlat; };
    const std::size_t N = static_cast<std::size_t>(nlat) * nlon;

    const double pct = std::clamp(percentile_, 0.0, 100.0) / 100.0;

    std::vector<int> outer(N, -1);
    std::vector<std::vector<int>> outer_cells;
    std::vector<int> stack;
    for (int j = 0; j < nlon; ++j) {
        for (int i = 0; i < nlat; ++i) {
            const int root = lin(i, j);
            if (outer[static_cast<std::size_t>(root)] != -1 || !on(i, j)) continue;
            const int cc_id = static_cast<int>(outer_cells.size());
            outer[static_cast<std::size_t>(root)] = cc_id;
            stack.clear();
            stack.push_back(root);
            std::vector<int> cells;
            while (!stack.empty()) {
                const int c = stack.back();
                stack.pop_back();
                cells.push_back(c);
                const int ci = c % nlat, cj = c / nlat;
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0) continue;
                        const int ii = ci + di, jj = cj + dj;
                        if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                        const int q = lin(ii, jj);
                        const std::size_t qz = static_cast<std::size_t>(q);
                        if (outer[qz] == -1 && on(ii, jj)) {
                            outer[qz] = cc_id;
                            stack.push_back(q);
                        }
                    }
            }
            outer_cells.push_back(std::move(cells));
        }
    }

    std::vector<int> cand;
    std::vector<double> cand_value;
    std::vector<char> visited(N, 0);
    std::vector<double> vals;

    for (std::size_t oc = 0; oc < outer_cells.size(); ++oc) {
        const auto& cells = outer_cells[oc];
        if (cells.empty()) continue;

        vals.clear();
        vals.reserve(cells.size());
        for (int c : cells) vals.push_back(Z(c % nlat, c / nlat));
        std::size_t kth = static_cast<std::size_t>(
            std::floor(pct * static_cast<double>(vals.size() - 1)));
        if (kth >= vals.size()) kth = vals.size() - 1;
        std::nth_element(vals.begin(), vals.begin() + kth, vals.end());
        const double pct_cutoff = vals[kth];
        const double floor_val = std::max(pct_cutoff, min_core_abs_);

        for (int c : cells) visited[static_cast<std::size_t>(c)] = 0;

        const int oc_id = static_cast<int>(oc);
        for (int seed_c : cells) {
            const std::size_t sz = static_cast<std::size_t>(seed_c);
            if (visited[sz]) continue;
            const int si = seed_c % nlat, sj = seed_c / nlat;
            if (Z(si, sj) < floor_val) continue;
            visited[sz] = 1;
            stack.clear();
            stack.push_back(seed_c);
            int peak = seed_c;
            double pv = Z(si, sj);
            int count = 0;
            while (!stack.empty()) {
                const int c = stack.back();
                stack.pop_back();
                ++count;
                const int ci = c % nlat, cj = c / nlat;
                if (Z(ci, cj) > pv) { pv = Z(ci, cj); peak = c; }
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0) continue;
                        const int ii = ci + di, jj = cj + dj;
                        if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                        const int q = lin(ii, jj);
                        const std::size_t qz = static_cast<std::size_t>(q);
                        if (!visited[qz] && outer[qz] == oc_id && Z(ii, jj) >= floor_val) {
                            visited[qz] = 1;
                            stack.push_back(q);
                        }
                    }
            }
            if (count >= n_min_) {
                cand.push_back(peak);
                cand_value.push_back(pv);
            }
        }
    }

    std::vector<int> order(cand.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return cand_value[a] > cand_value[b];
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

}
