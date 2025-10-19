cmake_minimum_required(VERSION 4.0)
project(Clustering LANGUAGES CXX)

# Require C++26
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic")

# Find dependencies via vcpkg's CONFIG packages
find_package(benchmark CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS math)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(Clustering
        src/main.cpp
)

# Public headers location
target_include_directories(Clustering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries (vcpkg provides imported targets benchmark::benchmark and fmt::fmt)
target_link_libraries(Clustering PRIVATE
        benchmark::benchmark
        fmt::fmt
        Boost::math
        Eigen3::Eigen
)

include(CTest)
enable_testing()

include(CPack)