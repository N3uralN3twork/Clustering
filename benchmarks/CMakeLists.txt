# benchmarks/CMakeLists.txt
# Build configuration for QuinnCluster performance benchmarks

cmake_minimum_required(VERSION 3.15)

# Only build benchmarks if explicitly requested
if(NOT BUILD_BENCHMARKS)
    return()
endif()

# Find Google Benchmark
find_package(benchmark CONFIG REQUIRED)

# ============================================================================
# Benchmark Executable
# ============================================================================

add_executable(qc_bench
        run_benchmarks.cpp
)

# Link against your library and Google Benchmark
target_link_libraries(qc_bench
        PRIVATE
        qc         # Your main library
        benchmark::benchmark
        benchmark::benchmark_main
)

# Include directories
target_include_directories(qc_bench
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Set compile options for maximum performance
if(MSVC)
    target_compile_options(qc_bench PRIVATE
            /O2              # Maximum optimization
            /Ob2             # Inline expansion
            /Oi              # Enable intrinsics
            /Ot              # Favor fast code
            /GL              # Whole program optimization
            /arch:AVX2       # Use AVX2 instructions (if available)
            /fp:fast         # Fast floating point
            /W4              # Warning level 4
    )

    target_link_options(qc_bench PRIVATE
            /LTCG            # Link-time code generation
    )
else()
    target_compile_options(qc_bench PRIVATE
            -O3
            -march=native
            -mtune=native
            -ffast-math
            -Wall
            -Wextra
    )
endif()

# ============================================================================
# Custom Benchmark Targets
# ============================================================================

# Quick benchmark: runs fast benchmarks only
add_custom_target(bench-quick
        COMMAND qc_bench
        --benchmark_filter="Small"
        --benchmark_format=console
        --benchmark_color=true
        DEPENDS qc_bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running quick benchmarks (small datasets only)"
)

# Full benchmark: runs all benchmarks
add_custom_target(bench-full
        COMMAND qc_bench
        --benchmark_format=console
        --benchmark_color=true
        --benchmark_out=full_results.json
        --benchmark_out_format=json
        DEPENDS qc_bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running full benchmark suite"
)

# Comparison benchmark: useful for before/after comparisons
add_custom_target(bench-compare
        COMMAND qc_bench
        --benchmark_format=json
        --benchmark_out=comparison_results.json
        --benchmark_repetitions=10
        --benchmark_display_aggregates_only=true
        DEPENDS qc_bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running benchmarks for comparison (10 repetitions)"
)

# ============================================================================
# Messages
# ============================================================================

message(STATUS "Benchmark targets:")
message(STATUS "  qc_bench - Main benchmark executable")
message(STATUS "  bench-quick             - Run quick benchmarks")
message(STATUS "  bench-full              - Run full benchmark suite")
message(STATUS "  bench-compare           - Run benchmarks for comparison")