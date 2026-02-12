#include "brew/plot_utils/plot_options.hpp"
#include <filesystem>
#include <thread>
#include <chrono>

namespace brew::plot_utils {

void save_figure(matplot::figure_handle fig, const std::string& filename) {
    if (filename.empty()) return;

    // Create parent directory if needed
    auto parent = std::filesystem::path(filename).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    fig->save(filename);

    // gnuplot runs asynchronously via a pipe. After flushing commands,
    // gnuplot still needs time to render and write the output file.
    // Wait for the file to appear on disk (up to 2 seconds).
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

} // namespace brew::plot_utils

