// benchmarks/clustering_benchmarks.cpp
// Example Google Benchmark file for QuinnCluster
//
// This file demonstrates how to structure benchmarks that align with
// the performance baselines from your Python implementation.

#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <Eigen/Dense>

// Include your clustering headers
// #include <quinncluster/algorithms/kmeans.hpp>
// #include <quinncluster/algorithms/hierarchical.hpp>
// #include <quinncluster/core/dataset.hpp>

// ============================================================================
// Helper Functions
// ============================================================================

// Generate random dataset for benchmarking
Eigen::MatrixXd GenerateRandomData(int n_observations, int n_variables, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<> dist(0.0, 1.0);
    
    Eigen::MatrixXd data(n_observations, n_variables);
    for (int i = 0; i < n_observations; ++i) {
        for (int j = 0; j < n_variables; ++j) {
            data(i, j) = dist(gen);
        }
    }
    return data;
}

// ============================================================================
// K-Means Benchmarks
// ============================================================================

// Small dataset: 1000 observations, 5 variables
// Python baseline: ~18 seconds, 63 MB
// C++ target: < 2 seconds, < 30 MB
static void BM_KMeans_Small(benchmark::State& state) {
    const int n_obs = 1000;
    const int n_vars = 5;
    const int k = 3;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // Replace with your actual implementation:
        // quinncluster::KMeans kmeans(k);
        // auto labels = kmeans.fit_predict(data);
        
        // Placeholder to prevent optimization
        benchmark::DoNotOptimize(data);
    }
    
    // Report theoretical complexity
    state.SetComplexityN(n_obs * n_vars * k);
}
BENCHMARK(BM_KMeans_Small)->Unit(benchmark::kMillisecond);

// Medium dataset: 5000 observations, 5 variables
// Python baseline: ~5.2 minutes (313 seconds), 2.6 GB
// C++ target: < 30 seconds, < 1 GB
static void BM_KMeans_Medium(benchmark::State& state) {
    const int n_obs = 5000;
    const int n_vars = 5;
    const int k = 3;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // quinncluster::KMeans kmeans(k);
        // auto labels = kmeans.fit_predict(data);
        
        benchmark::DoNotOptimize(data);
    }
    
    state.SetComplexityN(n_obs * n_vars * k);
}
BENCHMARK(BM_KMeans_Medium)->Unit(benchmark::kMillisecond);

// Large dataset: 10000 observations, 10 variables
// Python baseline: 8-18 minutes (480-1080 seconds), 3-8 GB
// C++ target: < 1 minute, < 2.5 GB
static void BM_KMeans_Large(benchmark::State& state) {
    const int n_obs = 10000;
    const int n_vars = 10;
    const int k = 3;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // quinncluster::KMeans kmeans(k);
        // auto labels = kmeans.fit_predict(data);
        
        benchmark::DoNotOptimize(data);
    }
    
    state.SetComplexityN(n_obs * n_vars * k);
}
BENCHMARK(BM_KMeans_Large)->Unit(benchmark::kMillisecond);

// Parameterized benchmark for different K values
static void BM_KMeans_VaryingK(benchmark::State& state) {
    const int n_obs = 1000;
    const int n_vars = 5;
    const int k = state.range(0);  // K varies
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // quinncluster::KMeans kmeans(k);
        // auto labels = kmeans.fit_predict(data);
        
        benchmark::DoNotOptimize(data);
    }
}
BENCHMARK(BM_KMeans_VaryingK)
    ->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(7)->Arg(10)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Hierarchical Clustering Benchmarks
// ============================================================================

static void BM_Hierarchical_Small(benchmark::State& state) {
    const int n_obs = 1000;
    const int n_vars = 5;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // quinncluster::Hierarchical hier(linkage::complete);
        // auto labels = hier.fit_predict(data, n_clusters=3);
        
        benchmark::DoNotOptimize(data);
    }
}
BENCHMARK(BM_Hierarchical_Small)->Unit(benchmark::kMillisecond);

// ============================================================================
// DBSCAN Benchmarks
// ============================================================================

static void BM_DBSCAN_Small(benchmark::State& state) {
    const int n_obs = 1000;
    const int n_vars = 5;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // quinncluster::DBSCAN dbscan(eps=0.5, min_samples=5);
        // auto labels = dbscan.fit_predict(data);
        
        benchmark::DoNotOptimize(data);
    }
}
BENCHMARK(BM_DBSCAN_Small)->Unit(benchmark::kMillisecond);

// ============================================================================
// Internal Metrics Benchmarks
// ============================================================================

// Benchmark Silhouette Score computation
static void BM_Silhouette_Score(benchmark::State& state) {
    const int n_obs = state.range(0);
    const int n_vars = 5;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    std::vector<int> labels(n_obs, 0);
    
    // Create some cluster structure
    for (int i = 0; i < n_obs; ++i) {
        labels[i] = i % 3;  // 3 clusters
    }
    
    for (auto _ : state) {
        // double score = quinncluster::metrics::silhouette_score(data, labels);
        
        benchmark::DoNotOptimize(labels);
    }
    
    state.SetComplexityN(n_obs);
}
BENCHMARK(BM_Silhouette_Score)
    ->RangeMultiplier(2)
    ->Range(100, 10000)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

// Benchmark Davies-Bouldin Index
static void BM_DaviesBouldin_Index(benchmark::State& state) {
    const int n_obs = state.range(0);
    const int n_vars = 5;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    std::vector<int> labels(n_obs, 0);
    
    for (int i = 0; i < n_obs; ++i) {
        labels[i] = i % 3;
    }
    
    for (auto _ : state) {
        // double score = quinncluster::metrics::davies_bouldin_index(data, labels);
        
        benchmark::DoNotOptimize(labels);
    }
}
BENCHMARK(BM_DaviesBouldin_Index)
    ->Arg(1000)->Arg(5000)->Arg(10000)
    ->Unit(benchmark::kMillisecond);

// Benchmark Calinski-Harabasz Index
static void BM_CalinskiHarabasz_Index(benchmark::State& state) {
    const int n_obs = state.range(0);
    const int n_vars = 5;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    std::vector<int> labels(n_obs, 0);
    
    for (int i = 0; i < n_obs; ++i) {
        labels[i] = i % 3;
    }
    
    for (auto _ : state) {
        // double score = quinncluster::metrics::calinski_harabasz_score(data, labels);
        
        benchmark::DoNotOptimize(labels);
    }
}
BENCHMARK(BM_CalinskiHarabasz_Index)
    ->Arg(1000)->Arg(5000)->Arg(10000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// External Metrics Benchmarks  
// ============================================================================

// Benchmark Rand Index
static void BM_Rand_Index(benchmark::State& state) {
    const int n_obs = state.range(0);
    
    std::vector<int> true_labels(n_obs);
    std::vector<int> pred_labels(n_obs);
    
    for (int i = 0; i < n_obs; ++i) {
        true_labels[i] = i % 3;
        pred_labels[i] = i % 4;
    }
    
    for (auto _ : state) {
        // double score = quinncluster::metrics::rand_index(true_labels, pred_labels);
        
        benchmark::DoNotOptimize(true_labels);
        benchmark::DoNotOptimize(pred_labels);
    }
}
BENCHMARK(BM_Rand_Index)
    ->Arg(1000)->Arg(5000)->Arg(10000)
    ->Unit(benchmark::kMillisecond);

// Benchmark Fowlkes-Mallows Index
static void BM_FowlkesMallows_Index(benchmark::State& state) {
    const int n_obs = state.range(0);
    
    std::vector<int> true_labels(n_obs);
    std::vector<int> pred_labels(n_obs);
    
    for (int i = 0; i < n_obs; ++i) {
        true_labels[i] = i % 3;
        pred_labels[i] = i % 4;
    }
    
    for (auto _ : state) {
        // double score = quinncluster::metrics::fowlkes_mallows_score(true_labels, pred_labels);
        
        benchmark::DoNotOptimize(true_labels);
    }
}
BENCHMARK(BM_FowlkesMallows_Index)
    ->Arg(1000)->Arg(5000)->Arg(10000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Full Pipeline Benchmarks (Clustering + All Fast Metrics)
// ============================================================================

// This matches your Python baseline: clustering + all fast metrics
static void BM_FullPipeline_Small(benchmark::State& state) {
    const int n_obs = 1000;
    const int n_vars = 5;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // 1. Perform clustering
        // quinncluster::KMeans kmeans(3);
        // auto labels = kmeans.fit_predict(data);
        
        // 2. Calculate all fast internal metrics
        // auto silhouette = quinncluster::metrics::silhouette_score(data, labels);
        // auto db_index = quinncluster::metrics::davies_bouldin_index(data, labels);
        // auto ch_index = quinncluster::metrics::calinski_harabasz_score(data, labels);
        // ... other fast metrics
        
        benchmark::DoNotOptimize(data);
    }
    
    // Compare against Python baseline: 18 seconds
    state.SetLabel("Python_baseline_18s");
}
BENCHMARK(BM_FullPipeline_Small)->Unit(benchmark::kSecond);

static void BM_FullPipeline_Large(benchmark::State& state) {
    const int n_obs = 10000;
    const int n_vars = 10;
    
    auto data = GenerateRandomData(n_obs, n_vars);
    
    for (auto _ : state) {
        // Full clustering + metrics pipeline
        
        benchmark::DoNotOptimize(data);
    }
    
    // Compare against Python baseline: 8-18 minutes
    state.SetLabel("Python_baseline_8-18min");
}
BENCHMARK(BM_FullPipeline_Large)->Unit(benchmark::kSecond);

// ============================================================================
// Memory Usage Benchmarks
// ============================================================================

// Benchmark to measure peak memory usage
static void BM_Memory_Large_Dataset(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto data = GenerateRandomData(10000, 10);
        state.ResumeTiming();
        
        // Run clustering
        // quinncluster::KMeans kmeans(3);
        // auto labels = kmeans.fit_predict(data);
        
        benchmark::DoNotOptimize(data);
    }
    
    // Expected: < 2.5 GB (Python uses 3-8 GB)
}
BENCHMARK(BM_Memory_Large_Dataset)->Unit(benchmark::kSecond);

// ============================================================================
// Main
// ============================================================================

BENCHMARK_MAIN();
