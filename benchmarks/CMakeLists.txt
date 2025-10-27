# ============================================================================
# Benchmark Executables
# ============================================================================

# Benchmark for utils functions
add_executable(benchmark_utils
        benchmark_utils.cpp
)

target_link_libraries(benchmark_utils
        PRIVATE
        clustering_lib
        benchmark::benchmark
        benchmark::benchmark_main
)

# Benchmark for internal metrics
add_executable(benchmark_internal
        benchmark_internal.cpp
)

target_link_libraries(benchmark_internal
        PRIVATE
        clustering_lib
        benchmark::benchmark
        benchmark::benchmark_main
)

# Set output directory for benchmarks
set_target_properties(benchmark_utils benchmark_internal
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/benchmarks"
)

# Add compiler optimizations for benchmarks
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(benchmark_utils PRIVATE -O3 -march=native)
    target_compile_options(benchmark_internal PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(benchmark_utils PRIVATE /O2)
    target_compile_options(benchmark_internal PRIVATE /O2)
endif()