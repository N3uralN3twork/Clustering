# tests/CMakeLists.txt
# Build configuration for QuinnCluster unit tests

cmake_minimum_required(VERSION 3.20)

# Only build tests if explicitly requested
if(NOT BUILD_TESTS)
    return()
endif()

# Find Google Test
find_package(GTest CONFIG REQUIRED)

# Enable testing
enable_testing()

# ============================================================================
# Test Executable
# ============================================================================

add_executable(clustering_tests
        # Add your test files here
        test_main.cpp
        # test_kmeans.cpp
        # test_hierarchical.cpp
        # test_dbscan.cpp
        # test_internal_metrics.cpp
        # test_external_metrics.cpp
        # test_dataset.cpp
)

# Link against your library and Google Test
target_link_libraries(clustering_tests
        PRIVATE
        clustering_lib           # Your main library
        GTest::gtest
        GTest::gtest_main
)

# Include directories
target_include_directories(clustering_tests
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Compiler options for tests
if(MSVC)
    target_compile_options(clustering_tests PRIVATE
            /W4              # Warning level 4
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(clustering_tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

# ============================================================================
# Register Tests with CTest
# ============================================================================

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(clustering_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        PROPERTIES
        TIMEOUT 60
)

# Or manually register tests
# add_test(NAME KMeansTests COMMAND clustering_tests --gtest_filter=KMeansTest.*)
# add_test(NAME MetricsTests COMMAND clustering_tests --gtest_filter=MetricsTest.*)

# ============================================================================
# Custom Test Targets
# ============================================================================

# Run tests with verbose output
add_custom_target(test-verbose
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure
        DEPENDS clustering_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running tests with verbose output"
)

# Run only fast tests
add_custom_target(test-fast
        COMMAND clustering_tests --gtest_filter=*Fast*
        DEPENDS clustering_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running fast tests only"
)

# Add after target creation:
target_compile_features(clustering_tests PRIVATE cxx_std_23)

# Update compiler options for MinGW:
if(MSVC)
    target_compile_options(clustering_tests PRIVATE /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(clustering_tests PRIVATE
            -Wall -Wextra -Wpedantic
            -std=c++23  # Explicit C++23
    )
endif()

# ============================================================================
# Messages
# ============================================================================

message(STATUS "Test targets configured:")
message(STATUS "  clustering_tests  - Main test executable")
message(STATUS "  test              - Run all tests (ctest)")
message(STATUS "  test-verbose      - Run tests with verbose output")
message(STATUS "  test-fast         - Run only fast tests")