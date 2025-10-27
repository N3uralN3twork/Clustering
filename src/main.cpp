#include <fmt/core.h>
#include <fmt/color.h> // For colored/bold text
#include <Eigen/Core>
#include <iostream>
#include <random>     // For synthetic data generation
#include <vector>     // For storing synthetic centers
#include <algorithm>  // For std::min, std::min_element, std::max_element
#include <iterator>   // For std::distance
#include <string>     // For std::string
#include <functional> // For std::function

#include "qc/internal/metrics.hpp"

// Define our data types for convenience
using Matrixd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Vectord = Eigen::Vector<double, Eigen::Dynamic>;
using Vectori = Eigen::VectorXi;

/**
 * @brief Generates a synthetic "blob" dataset.
 * (Unchanged from previous version)
 */
std::pair<Matrixd, Vectori> generate_synthetic_data(
    int n_samples, int n_features, int n_clusters, double cluster_std,
    std::mt19937& rng // Use a reference to the generator
) {
    Matrixd data = Matrixd::Zero(n_samples, n_features);
    Vectori labels = Vectori::Zero(n_samples);

    // --- Random Number Generation Setup ---
    std::uniform_real_distribution<double> center_dist(-50.0, 50.0);
    std::normal_distribution<double> point_dist(0.0, cluster_std);

    // --- 1. Create true cluster centers ---
    std::vector<Vectord> centers;
    for (int k = 0; k < n_clusters; ++k) {
        Vectord center(n_features);
        for (int j = 0; j < n_features; ++j) {
            center(j) = center_dist(rng);
        }
        centers.push_back(center);
    }

    // --- 2. Generate points for each cluster ---
    int samples_per_cluster = n_samples / n_clusters;
    for (int i = 0; i < n_samples; ++i) {
        int cluster_id = std::min(i / samples_per_cluster, n_clusters - 1);
        labels(i) = cluster_id;

        for (int j = 0; j < n_features; ++j) {
            data(i, j) = centers[cluster_id](j) + point_dist(rng);
        }
    }

    return {data, labels};
}

// --- NEW: Define the criterion enum ---
enum class Criterion {
    Minimize,
    Maximize
    // We could add "FindElbow", "FindMaxDiff", etc. later
};

// --- NEW: Define the struct to manage a single metric's test run ---
struct MetricRunner {
    // --- Parameters (set at creation) ---
    std::string name;
    Criterion criterion;
    std::function<double(const Matrixd&, const Vectori&)> metric_func;

    // --- State (filled during run) ---
    std::vector<double> results;
    int best_k = 0;

    /**
     * @brief Runs the stored metric function on the data and stores the result.
     */
    void run(const Matrixd& data, const Vectori& labels) {
        results.push_back(metric_func(data, labels));
    }

    /**
     * @brief Analyzes the 'results' vector to find the best k based on the criterion.
     */
    void find_best_k(int min_clusters) {
        if (results.empty()) {
            best_k = -1; // Error or no data
            return;
        }

        if (criterion == Criterion::Minimize) {
            auto it = std::min_element(results.begin(), results.end());
            best_k = std::distance(results.begin(), it) + min_clusters;
        } else { // Criterion::Maximize
            auto it = std::max_element(results.begin(), results.end());
            best_k = std::distance(results.begin(), it) + min_clusters;
        }
    }

    /**
     * @brief Prints this metric's full row for the results table.
     */
    void print_row(int name_w, int crit_w, int best_k_w) const {
        std::string crit_str = (criterion == Criterion::Minimize) ? "Low" : "High";

        fmt::print("{:<{}} | {:<{}} | {:<{}} |",
            name, name_w, crit_str, crit_w, fmt::format("k={}", best_k), best_k_w);

        for (double val : results) {
            fmt::print(" {:>12.2f} |", val);
        }
        fmt::print("\n");
    }
};


int main() {
    fmt::print("Testing clustering metrics on synthetic data...\n\n");

    // --- 1. Define synthetic dataset parameters ---
    constexpr int N_SAMPLES = 1'000;
    constexpr int N_FEATURES = 5;
    constexpr double CLUSTER_STD = 2.5;
    constexpr int MIN_CLUSTERS = 2;
    constexpr int MAX_CLUSTERS = 6;

    fmt::print("Dataset Parameters:\n");
    fmt::print("- Samples:    {}\n", N_SAMPLES);
    fmt::print("- Features:   {}\n", N_FEATURES);
    fmt::print("- Std. Dev:   {}\n\n", CLUSTER_STD);

    // (Note on methodology unchanged)
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::yellow),
        "NOTE: ");
    fmt::print("This simulation generates a new dataset *with* 'k' clusters for each 'k' value.\n\n");


    // Use a single, fixed-seed generator for reproducible runs
    std::mt19937 rng(42);

    // --- 2. Data storage for results ---
    // NEW: Create a vector of MetricRunner structs.
    // This is the ONLY part to edit when adding new metrics.
    std::vector<MetricRunner> all_metrics;

    all_metrics.push_back({
        "Ball-Hall",
        Criterion::Minimize,
        qc::internal::Metrics::ball_hall_index<double> // Pass the function
    });

    all_metrics.push_back({
        "Calinski-Harabasz",
        Criterion::Maximize,
        qc::internal::Metrics::calinski_harabasz_index<double> // Pass the function
    });

    all_metrics.push_back({
        "Trace(W)",
        Criterion::Minimize,
        qc::internal::Metrics::trace_w_index<double> // Pass the function
    });


    // --- 3. Run all metrics for k in [min, max] ---
    fmt::print("Running simulation for k = {} to {}...\n", MIN_CLUSTERS, MAX_CLUSTERS);
    for (int k = MIN_CLUSTERS; k <= MAX_CLUSTERS; ++k) {
        // Generate new data for each k
        auto [data, labels] = generate_synthetic_data(
            N_SAMPLES, N_FEATURES, k, CLUSTER_STD, rng
        );

        // NEW: Simple inner loop to run all metrics
        for (auto& runner : all_metrics) {
            runner.run(data, labels);
        }
    }
    fmt::print("Simulation complete. Generating report...\n\n");

    // --- 4. Find the "Best k" for each metric ---
    // NEW: Simple loop to analyze all results
    for (auto& runner : all_metrics) {
        runner.find_best_k(MIN_CLUSTERS);
    }

    // --- 5. Print Table ---
    // Define column widths
    const int METRIC_WIDTH = 25;
    const int CRIT_WIDTH = 10;
    const int BEST_K_WIDTH = 8;
    const int K_WIDTH = 12; // Width for each "k=N" column

    // --- Header Row 1 (Titles) --- (Unchanged)
    auto header_style = fmt::emphasis::bold | fmt::fg(fmt::color::cyan);
    fmt::print(header_style, "{:<{}} | {:<{}} | {:<{}} |",
        "Metric", METRIC_WIDTH, "Criterion", CRIT_WIDTH, "Best k", BEST_K_WIDTH);
    for (int k = MIN_CLUSTERS; k <= MAX_CLUSTERS; ++k) {
        fmt::print(header_style, " {:>{}} |", fmt::format("k={}", k), K_WIDTH);
    }
    fmt::print("\n");

    // --- Header Row 2 (Separators) --- (Unchanged)
    fmt::print(header_style, "{}-|{}-|{}-|",
        std::string(METRIC_WIDTH + 1, '-'),
        std::string(CRIT_WIDTH + 1, '-'),
        std::string(BEST_K_WIDTH + 1, '-'));
    for (int k = MIN_CLUSTERS; k <= MAX_CLUSTERS; ++k) {
        fmt::print(header_style, "{}-|", std::string(K_WIDTH + 2, '-'));
    }
    fmt::print("\n");

    // --- NEW: Data Rows (Loop) ---
    for (const auto& runner : all_metrics) {
        runner.print_row(METRIC_WIDTH, CRIT_WIDTH, BEST_K_WIDTH);
    }

    // --- Footer Row (Separators) --- (Unchanged)
    fmt::print(header_style, "{}-|{}-|{}-|",
        std::string(METRIC_WIDTH + 1, '-'),
        std::string(CRIT_WIDTH + 1, '-'),
        std::string(BEST_K_WIDTH + 1, '-'));
    for (int k = MIN_CLUSTERS; k <= MAX_CLUSTERS; ++k) {
        fmt::print(header_style, "{}-|", std::string(K_WIDTH + 2, '-'));
    }
    fmt::print("\n");

    return 0;
}

