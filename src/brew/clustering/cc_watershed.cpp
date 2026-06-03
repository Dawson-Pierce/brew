#include "brew/clustering/cc_watershed.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brew::clustering {

CCWatershed::CCWatershed(std::vector<double> lon, std::vector<double> lat,
                         int closing_radius, int min_seed_dist,
                         int min_size, double min_core)
    : lon_(std::move(lon)), lat_(std::move(lat)),
      closing_radius_(closing_radius), min_seed_dist_(min_seed_dist),
      min_size_(min_size), min_core_(min_core) {}

std::vector<Eigen::MatrixXd> CCWatershed::cluster(const Eigen::MatrixXd& Z) const {
    const int nlat = static_cast<int>(Z.rows());
    const int nlon = static_cast<int>(Z.cols());
    if (nlat == 0 || nlon == 0) return {};

    auto on = [&](int i, int j) {
        const double v = Z(i, j);
        return std::isfinite(v) && v > 0.0;
    };
    auto lin = [nlat](int i, int j) { return i + j * nlat; };
    const std::size_t N = static_cast<std::size_t>(nlat) * nlon;

    const int r_close = std::max(closing_radius_, 0);

    std::vector<char> in_dilated(N, 0);
    if (r_close == 0) {
        for (int j = 0; j < nlon; ++j)
            for (int i = 0; i < nlat; ++i)
                if (on(i, j))
                    in_dilated[static_cast<std::size_t>(lin(i, j))] = 1;
    } else {
        for (int j = 0; j < nlon; ++j) {
            for (int i = 0; i < nlat; ++i) {
                if (!on(i, j)) continue;
                const int j1 = std::max(0, j - r_close), j2 = std::min(nlon - 1, j + r_close);
                const int i1 = std::max(0, i - r_close), i2 = std::min(nlat - 1, i + r_close);
                for (int jj = j1; jj <= j2; ++jj)
                    for (int ii = i1; ii <= i2; ++ii)
                        in_dilated[static_cast<std::size_t>(lin(ii, jj))] = 1;
            }
        }
    }

    std::vector<int> region(N, -1);
    int n_regions = 0;
    std::vector<int> stack;
    for (int j = 0; j < nlon; ++j) {
        for (int i = 0; i < nlat; ++i) {
            const int root = lin(i, j);
            const std::size_t rz = static_cast<std::size_t>(root);
            if (region[rz] != -1 || !in_dilated[rz]) continue;
            const int rid = n_regions++;
            region[rz] = rid;
            stack.clear();
            stack.push_back(root);
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
                        if (region[qz] != -1 || !in_dilated[qz]) continue;
                        region[qz] = rid;
                        stack.push_back(q);
                    }
            }
        }
    }

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

    std::vector<int> label(N, -1);
    std::vector<char> blocked(N, 0);
    std::vector<int> seeds;
    int nseed = 0;
    const int r_seed = std::max(min_seed_dist_, 0);
    for (int p : maxima) {
        if (blocked[static_cast<std::size_t>(p)]) continue;
        const int i = p % nlat, j = p / nlat;
        label[static_cast<std::size_t>(p)] = nseed++;
        seeds.push_back(p);
        const int j1 = std::max(0, j - r_seed), j2 = std::min(nlon - 1, j + r_seed);
        const int i1 = std::max(0, i - r_seed), i2 = std::min(nlat - 1, i + r_seed);
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
                if (!on(ii, jj) || label[qz] != -1 || queued[qz]) continue;
                if (region[qz] != region[static_cast<std::size_t>(cell)]) continue;
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
                if (region[static_cast<std::size_t>(q)] != region[static_cast<std::size_t>(cell)]) continue;
                basin = label[static_cast<std::size_t>(q)];
                break;
            }
        }
        if (basin < 0) continue;
        label[static_cast<std::size_t>(cell)] = basin;
        push_nbrs(cell);
    }

    std::vector<char> region_has_seed(static_cast<std::size_t>(n_regions), 0);
    for (int s : seeds) {
        const int rid = region[static_cast<std::size_t>(s)];
        if (rid >= 0) region_has_seed[static_cast<std::size_t>(rid)] = 1;
    }

    std::unordered_map<int, int> region_basin;
    for (int j = 0; j < nlon; ++j) {
        for (int i = 0; i < nlat; ++i) {
            const int c = lin(i, j);
            const std::size_t cz = static_cast<std::size_t>(c);
            if (!on(i, j) || label[cz] != -1) continue;
            const int rid = region[cz];
            if (rid < 0) continue;
            if (region_has_seed[static_cast<std::size_t>(rid)]) continue;
            int basin;
            auto it = region_basin.find(rid);
            if (it == region_basin.end()) {
                basin = nseed++;
                region_basin[rid] = basin;
            } else {
                basin = it->second;
            }
            stack.clear();
            stack.push_back(c);
            label[cz] = basin;
            while (!stack.empty()) {
                const int cc = stack.back();
                stack.pop_back();
                const int cci = cc % nlat, ccj = cc / nlat;
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0) continue;
                        const int ii = cci + di, jj = ccj + dj;
                        if (ii < 0 || ii >= nlat || jj < 0 || jj >= nlon) continue;
                        const int q = lin(ii, jj);
                        const std::size_t qz = static_cast<std::size_t>(q);
                        if (!on(ii, jj) || label[qz] != -1) continue;
                        if (region[qz] != rid) continue;
                        label[qz] = basin;
                        stack.push_back(q);
                    }
            }
        }
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
        double peak = -std::numeric_limits<double>::infinity();
        for (int p : cells) peak = std::max(peak, Z(p % nlat, p / nlat));
        if (peak < min_core_) continue;
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
