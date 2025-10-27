#include <benchmark/benchmark.h>
#include <fmt/core.h>

#include "qc/internal/metrics.hpp"

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

static void BM_ball_hall_index(benchmark::State& state) {
    const BenchmarkData test_data(state.range(0), state.range(1), state.range(2));

    for (auto _ : state) {
        auto result = qc::internal::Metrics::ball_hall_index<double>(test_data.data, test_data.labels);
        benchmark::DoNotOptimize(result);
    }

    state.counters["Rows"] = state.range(0);
    state.counters["Features"] = state.range(1);
    state.counters["Clusters"] = state.range(2);
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
    RegisterBenchmark("Ball-Hall Index", BM_ball_hall_index);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}