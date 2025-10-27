#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include "qc/utils/utils.hpp"

// Shared test data generator
struct BenchmarkData {
    Eigen::MatrixXd data;
    Eigen::VectorXi labels;

    BenchmarkData(const long num_samples, const long num_features, const int num_clusters) {
        data = Eigen::MatrixXd::Random(num_samples, num_features);
        labels = (Eigen::ArrayXd::Random(num_samples).abs() * num_clusters)
                     .floor().cast<int>()
                     .cwiseMin(num_clusters - 1).cwiseMax(0);
    }
};

static void BM_get_centroids(benchmark::State& state) {
    const BenchmarkData test_data(state.range(0), state.range(1), state.range(2));

    for (auto _ : state) {
        auto result = qc::utils::get_centroids<double>(test_data.data, test_data.labels);
        benchmark::DoNotOptimize(result.data());
    }

    state.counters["Num. Rows"] = state.range(0);
    state.counters["Num. Features"] = state.range(1);
    state.counters["Num. Clusters"] = state.range(2);
}

static void BM_GSS(benchmark::State& state) {
    const BenchmarkData test_data(state.range(0), state.range(1), state.range(2));

    for (auto _ : state) {
        auto result = qc::utils::gss<double>(test_data.data, test_data.labels);
        benchmark::DoNotOptimize(result);
    }

    state.counters["Num. Rows"] = state.range(0);
    state.counters["Num. Feats"] = state.range(1);
    state.counters["Num. Clusters"] = state.range(2);
}

static void BM_get_per_cluster_wgss(benchmark::State& state) {
    const BenchmarkData test_data(state.range(0), state.range(1), state.range(2));
    const auto centroids = qc::utils::get_centroids<double>(test_data.data, test_data.labels);
    for (auto _ : state) {
        Eigen::Vector<double, Eigen::Dynamic> result = qc::utils::get_per_cluster_wgss<double>(test_data.data, test_data.labels, centroids);
        benchmark::DoNotOptimize(result);
    }

    state.counters["Num. Rows"] = state.range(0);
    state.counters["Num. Feats"] = state.range(1);
    state.counters["Num. Clusters"] = state.range(2);
}


// Helper to register benchmarks with standard arguments
template<typename Func>
void RegisterBenchmark(const char* name, Func* func) {
    benchmark::RegisterBenchmark(name, func)
        ->Unit(benchmark::kMillisecond)
        ->Args({1000, 10, 5})
        ->Args({10000, 10, 5})
        ->Args({10000, 100, 5})
        ->Args({100000, 10, 5})
        ->Args({100000, 100, 50})
        ->Args({1000000, 100, 50});
}

int main(int argc, char** argv) {
    RegisterBenchmark("Centroids", BM_get_centroids);
    RegisterBenchmark("Sum Squares", BM_GSS);
    RegisterBenchmark("Tot. Cluster Sep.", BM_get_per_cluster_wgss);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}