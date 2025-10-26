#include <benchmark/benchmark.h>
#include "qc/utils/utils.hpp"

/**
 * @brief Benchmark fixture for the Utils::centers2 function.
 *
 * This function generates test data based on benchmark arguments and
 * times the execution of Utils::centers2.
 *
 * The `state` object is managed by Google Benchmark.
 * `state.range(0)` = Number of Samples
 * `state.range(1)` = Number of Features
 * `state.range(2)` = Number of Clusters
 */
static void BM_Centers2(benchmark::State& state) {

    // --- Setup ---
    // This code runs ONCE per benchmark, NOT during timing.
    const long num_samples = state.range(0);
    const long num_features = state.range(1);
    const int num_clusters = state.range(2);

    // 1. Create a random data matrix (n_samples x n_features)
    //    Values are between -1 and 1.
    const Eigen::MatrixXd data = Eigen::MatrixXd::Random(num_samples, num_features);

    // 2. Create a random label vector (n_samples x 1)
    //    We use Eigen's random array capabilities to get integers
    //    in the range [0, num_clusters - 1].
    Eigen::VectorXi labels = (Eigen::ArrayXd::Random(num_samples)
                                .abs() * num_clusters)
                                .floor()
                                .cast<int>();

    // Ensure labels are clipped correctly, just in case
    labels = labels.cwiseMin(num_clusters - 1).cwiseMax(0);


    // --- Timing Loop ---
    // This loop is timed by Google Benchmark.
    // It will run multiple iterations to get a stable measurement.
    for (auto _ : state) {
        // Call the function we want to test
        Eigen::MatrixXd result = Utils::centers2(data, labels);

        // Prevent the compiler from optimizing away the function call
        // because `result` isn't "used".
        benchmark::DoNotOptimize(result.data());
    }

    // Optional: Set a custom label for the benchmark output
    state.SetLabel(
        std::to_string(num_samples) + "s_" +
        std::to_string(num_features) + "f_" +
        std::to_string(num_clusters) + "k"
    );
}

// --- Register the Benchmark ---
// We define several test cases with different parameters.
// Format is: .Args({num_samples, num_features, num_clusters})

BENCHMARK(BM_Centers2)
    ->Unit(benchmark::kMillisecond)
    ->Args({1000, 10, 5})       // 1k samples, 10 features, 5 clusters
    ->Args({10000, 10, 5})      // 10k samples, 10 features, 5 clusters
    ->Args({10000, 100, 5})     // 10k samples, 100 features, 5 clusters
    ->Args({100000, 10, 5})     // 100k samples, 10 features, 5 clusters
    ->Args({100000, 100, 50})   // 100k samples, 100 features, 50 clusters
    ->Args({1000000, 100, 50}); // 1M samples, 100 features, 50 clusters
