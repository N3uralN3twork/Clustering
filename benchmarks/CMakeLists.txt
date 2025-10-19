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

add_executable(quinncluster_benchmarks
    clustering_benchmarks.cpp
)

# Link against your library and Google Benchmark
target_link_libraries(quinncluster_benchmarks
    PRIVATE
        quinncluster_lib         # Your main library
        benchmark::benchmark
        benchmark::benchmark_main
)

# Include directories
target_include_directories(quinncluster_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Set compile options for maximum performance
if(MSVC)
    target_compile_options(quinncluster_benchmarks PRIVATE
        /O2              # Maximum optimization
        /Ob2             # Inline expansion
        /Oi              # Enable intrinsics
        /Ot              # Favor fast code
        /GL              # Whole program optimization
        /arch:AVX2       # Use AVX2 instructions (if available)
        /fp:fast         # Fast floating point
        /W4              # Warning level 4
    )
    
    target_link_options(quinncluster_benchmarks PRIVATE
        /LTCG            # Link-time code generation
    )
else()
    target_compile_options(quinncluster_benchmarks PRIVATE
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -Wall
        -Wextra
    )
endif()

# ============================================================================
# Additional Benchmark Executables (Optional)
# ============================================================================

# Separate benchmark for algorithms only
add_executable(algorithm_benchmarks
    algorithm_benchmarks.cpp
)

target_link_libraries(algorithm_benchmarks
    PRIVATE
        quinncluster_lib
        benchmark::benchmark
        benchmark::benchmark_main
)

target_include_directories(algorithm_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(algorithm_benchmarks PRIVATE /O2 /arch:AVX2)
endif()

# Separate benchmark for metrics only
add_executable(metrics_benchmarks
    metrics_benchmarks.cpp
)

target_link_libraries(metrics_benchmarks
    PRIVATE
        quinncluster_lib
        benchmark::benchmark
        benchmark::benchmark_main
)

target_include_directories(metrics_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(metrics_benchmarks PRIVATE /O2 /arch:AVX2)
endif()

# ============================================================================
# Installation
# ============================================================================

# Don't install benchmarks by default
# install(TARGETS quinncluster_benchmarks DESTINATION bin OPTIONAL)

# ============================================================================
# Custom Benchmark Targets
# ============================================================================

# Quick benchmark: runs fast benchmarks only
add_custom_target(bench-quick
    COMMAND quinncluster_benchmarks
        --benchmark_filter="Small"
        --benchmark_format=console
        --benchmark_color=true
    DEPENDS quinncluster_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running quick benchmarks (small datasets only)"
)

# Full benchmark: runs all benchmarks
add_custom_target(bench-full
    COMMAND quinncluster_benchmarks
        --benchmark_format=console
        --benchmark_color=true
        --benchmark_out=full_results.json
        --benchmark_out_format=json
    DEPENDS quinncluster_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running full benchmark suite"
)

# Comparison benchmark: useful for before/after comparisons
add_custom_target(bench-compare
    COMMAND quinncluster_benchmarks
        --benchmark_format=json
        --benchmark_out=comparison_results.json
        --benchmark_repetitions=10
        --benchmark_display_aggregates_only=true
    DEPENDS quinncluster_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks for comparison (10 repetitions)"
)

# Memory profiling benchmark (if available)
if(WIN32)
    add_custom_target(bench-memory
        COMMAND drmemory.exe -light -- quinncluster_benchmarks
            --benchmark_filter="Memory"
            --benchmark_repetitions=1
        DEPENDS quinncluster_benchmarks
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running memory profiling benchmarks"
    )
endif()

# ============================================================================
# Messages
# ============================================================================

message(STATUS "Benchmark targets:")
message(STATUS "  quinncluster_benchmarks - Main benchmark executable")
message(STATUS "  bench-quick             - Run quick benchmarks")
message(STATUS "  bench-full              - Run full benchmark suite")
message(STATUS "  bench-compare           - Run benchmarks for comparison")
if(WIN32)
    message(STATUS "  bench-memory            - Run memory profiling (requires Dr. Memory)")
endif()
