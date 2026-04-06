add_executable(NonCooperativeSpacecraftExample
    NonCooperativeSpacecraftExample.cpp
)

target_link_libraries(NonCooperativeSpacecraftExample PRIVATE
    brew
    yaml-cpp::yaml-cpp
)

# Copy YAML config next to the binary isn't needed — we run from project root.
# The executable expects: examples/NonCooperativeSpacecraftConfig.yaml
# (or pass a custom path as argv[1]).
