add_executable(NonCooperativeSpacecraftExample
    NonCooperativeSpacecraftExample.cpp
)

target_link_libraries(NonCooperativeSpacecraftExample PRIVATE
    brew
)

# All configuration is inlined as a C++ Config struct in the example source —
# edit NonCooperativeSpacecraftExample.cpp and rebuild to change parameters.
# The example expects STL template files (e.g. A.stl, B.stl) in the working
# directory at runtime. Optional: pass argv[1] to override num_steps.
