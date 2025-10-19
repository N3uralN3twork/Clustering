cmake_minimum_required(VERSION 3.15)
project(Clustering VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Build Options
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Note: C++26 is very cutting-edge. Consider C++20 or C++23 for better compatibility
# set(CMAKE_CXX_STANDARD 20)  # More widely supported

# ============================================================================
# vcpkg Configuration
# ============================================================================
# Set triplet based on compiler
if(MINGW)
    set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic" CACHE STRING "vcpkg triplet")
elseif(MSVC)
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg triplet")
endif()

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(fmt CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS math)
find_package(Eigen3 CONFIG REQUIRED)

# Conditionally find test/benchmark dependencies
if(BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)
endif()

if(BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
endif()

# ============================================================================
# Library Target (Core Clustering Algorithms)
# ============================================================================
# This is crucial - your algorithms should be in a library that can be
# linked by the main executable, tests, and benchmarks

add_library(clustering_lib STATIC
        # Add your source files here as you create them
        # src/algorithms/kmeans.cpp
        # src/algorithms/hierarchical.cpp
        # src/algorithms/dbscan.cpp
        # src/metrics/internal_metrics.cpp
        # src/metrics/external_metrics.cpp
        # src/core/dataset.cpp
        # etc.
)

# Public headers location
target_include_directories(clustering_lib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies to library
target_link_libraries(clustering_lib
        PUBLIC
        Eigen3::Eigen
        fmt::fmt
        Boost::math
)

# Compiler-specific optimizations
if(MSVC)
    target_compile_options(clustering_lib PRIVATE
            /W4          # Warning level 4
            $<$<CONFIG:Release>:/O2>  # Optimize for speed
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(clustering_lib PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
    )
endif()

# ============================================================================
# Main Executable
# ============================================================================
add_executable(Clustering
        src/main.cpp
)

target_link_libraries(Clustering
        PRIVATE
        clustering_lib
)

# ============================================================================
# Tests
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Examples (Optional)
# ============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Installation (Optional)
# ============================================================================
include(GNUInstallDirs)

install(TARGETS clustering_lib Clustering
        EXPORT ClusteringTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ============================================================================
# CPack Configuration
# ============================================================================
include(CPack)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

# ============================================================================
# Print Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "=============================================================")
message(STATUS "  ${PROJECT_NAME} v${PROJECT_VERSION} Configuration Summary")
message(STATUS "=============================================================")
message(STATUS "  CMake version          : ${CMAKE_VERSION}")
message(STATUS "  Compiler               : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard           : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type             : ${CMAKE_BUILD_TYPE}")
message(STATUS "  vcpkg triplet          : ${VCPKG_TARGET_TRIPLET}")
message(STATUS "")
message(STATUS "  Build options:")
message(STATUS "    BUILD_TESTS          : ${BUILD_TESTS}")
message(STATUS "    BUILD_BENCHMARKS     : ${BUILD_BENCHMARKS}")
message(STATUS "    BUILD_EXAMPLES       : ${BUILD_EXAMPLES}")
message(STATUS "")
message(STATUS "  Dependencies:")
message(STATUS "    Eigen3               : ${Eigen3_VERSION}")
message(STATUS "    fmt                  : ${fmt_VERSION}")
message(STATUS "    Boost                : ${Boost_VERSION}")
if(BUILD_BENCHMARKS)
    message(STATUS "    Google Benchmark     : Found")
endif()
if(BUILD_TESTS)
    message(STATUS "    Google Test          : Found")
endif()
message(STATUS "=============================================================")
message(STATUS "")