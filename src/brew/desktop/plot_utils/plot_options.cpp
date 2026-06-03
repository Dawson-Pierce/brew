#include "brew/desktop/plot_utils/plot_options.hpp"
#include <filesystem>
#include <thread>
#include <chrono>

namespace brew::plot_utils {

void save_figure(matplot::figure_handle fig, const std::string& filename) {
    if (filename.empty()) return;

    auto parent = std::filesystem::path(filename).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    fig->save(filename);

    using namespace std::chrono;
    auto deadline = steady_clock::now() + seconds(2);
    while (steady_clock::now() < deadline) {
        if (std::filesystem::exists(filename) &&
            std::filesystem::file_size(filename) > 0) {
            return;
        }
        std::this_thread::sleep_for(milliseconds(10));
    }
}

}
