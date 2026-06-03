#include "brew/clustering/grid_cc.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace brew::clustering {

GridCC::GridCC(std::vector<double> lon, std::vector<double> lat,
               int closing_radius, int min_size)
    : lon_(std::move(lon)), lat_(std::move(lat)),
      closing_radius_(closing_radius), min_size_(min_size) {}

std::vector<Eigen::MatrixXd> GridCC::cluster(const Eigen::MatrixXd& Z) const {
    const int nlat = static_cast<int>(Z.rows());
    const int nlon = static_cast<int>(Z.cols());
    if (nlat == 0 || nlon == 0) return {};

    auto on = [&](int i, int j) {
        const double v = Z(i, j);
        return std::isfinite(v) && v > 0.0;
    };
    auto lin = [nlat](int i, int j) { return i + j * nlat; };

    std::vector<int> parent(static_cast<std::size_t>(nlat) * nlon, -1);
    for (int j = 0; j < nlon; ++j)
        for (int i = 0; i < nlat; ++i)
            if (on(i, j)) parent[lin(i, j)] = lin(i, j);

    auto find = [&parent](int x) {
        int root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != root) { int nx = parent[x]; parent[x] = root; x = nx; }
        return root;
    };
    auto unite = [&](int a, int b) {
        const int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    };

    const int r = closing_radius_ + 1;
    for (int j = 0; j < nlon; ++j) {
        for (int i = 0; i < nlat; ++i) {
            if (!on(i, j)) continue;
            const int p = lin(i, j);
            const int j1 = std::max(0, j - r), j2 = std::min(nlon - 1, j + r);
            const int i1 = std::max(0, i - r), i2 = std::min(nlat - 1, i + r);
            for (int jj = j1; jj <= j2; ++jj)
                for (int ii = i1; ii <= i2; ++ii)
                    if ((ii != i || jj != j) && on(ii, jj))
                        unite(p, lin(ii, jj));
        }
    }

    std::unordered_map<int, std::vector<int>> groups;
    for (int j = 0; j < nlon; ++j)
        for (int i = 0; i < nlat; ++i)
            if (on(i, j)) groups[find(lin(i, j))].push_back(lin(i, j));

    const bool have_lon = static_cast<int>(lon_.size()) == nlon;
    const bool have_lat = static_cast<int>(lat_.size()) == nlat;

    std::vector<Eigen::MatrixXd> result;
    for (const auto& kv : groups) {
        const std::vector<int>& cells = kv.second;
        if (static_cast<int>(cells.size()) < min_size_) continue;
        Eigen::MatrixXd C(3, static_cast<Eigen::Index>(cells.size()));
        for (std::size_t w = 0; w < cells.size(); ++w) {
            const int p = cells[w];
            const int i = p % nlat;
            const int j = p / nlat;
            const auto col = static_cast<Eigen::Index>(w);
            C(0, col) = have_lon ? lon_[j] : static_cast<double>(j);
            C(1, col) = have_lat ? lat_[i] : static_cast<double>(i);
            C(2, col) = Z(i, j);
        }
        result.push_back(std::move(C));
    }
    return result;
}

}
