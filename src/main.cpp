#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <string>
#include <functional>
#include <algorithm>
#include <chrono>
#include <map>
#include <Eigen/Dense>
#include <tabulate/table.hpp>
#include <fmt/core.h>

// Include your custom headers
#include "qc/internal/metrics.hpp"

using namespace tabulate;

// ============================================================================
// Metric Registry System
// ============================================================================

/**
 * @brief Enum for metric optimization direction
 */
enum class MetricDirection {
    LOWER_IS_BETTER,
    HIGHER_IS_BETTER
};

/**
 * @brief Enum for metric category
 */
enum class MetricCategory {
    INTERNAL,  // Does not require ground truth labels
    EXTERNAL   // Requires ground truth labels
};

/**
 * @brief Type alias for metric function
 */
using MetricFunction = std::function<double(
    const Eigen::Ref<const Eigen::MatrixXd>&,
    const Eigen::Ref<const Eigen::VectorXi>&
)>;

/**
 * @brief Struct to hold metric metadata and configuration
 */
struct MetricInfo {
    std::string name;
    MetricFunction function;
    MetricDirection direction;
    MetricCategory category;
    bool enabled;

    MetricInfo(
        std::string  name,
        MetricFunction func,
        const MetricDirection dir,
        const MetricCategory cat,
        const bool e = true
    ) : name(std::move(name)), function(std::move(func)), direction(dir), category(cat), enabled(e) {}
};

/**
 * @brief Metric Registry - stores all available metrics
 */
class MetricRegistry {
private:
    std::vector<MetricInfo> metrics_;

public:
    void add_metric(const MetricInfo& metric) {
        metrics_.push_back(metric);
    }

    std::vector<MetricInfo>& get_metrics() {
        return metrics_;
    }

    const std::vector<MetricInfo>& get_metrics() const {
        return metrics_;
    }

    void sort_alphabetically() {
        std::sort(metrics_.begin(), metrics_.end(),
                  [](const MetricInfo& a, const MetricInfo& b) {
                      return a.name < b.name;
                  });
    }

    size_t enabled_count() const {
        return std::count_if(metrics_.begin(), metrics_.end(),
                            [](const MetricInfo& m) { return m.enabled; });
    }
};

// ============================================================================
// Configuration Parameters
// ============================================================================
struct Config {
    // Data generation parameters
    int MIN_CLUSTERS = 2;
    int MAX_CLUSTERS = 8;
    int N_SAMPLES = 5'000;
    int N_FEATURES = 4;
    double CLUSTER_STD = 0.8;
    int TRUE_CLUSTERS = 4;  // Ground truth number of clusters
    int RANDOM_SEED = 421;

    // K-means parameters
    int MAX_KMEANS_ITER = 200;
    double KMEANS_TOL = 1e-4;

    // Metric enable/disable flags
    struct MetricFlags {
        // Internal metrics (add more as you implement them)
        bool ball_hall = true;
        bool calinski_harabasz = true;
        bool trace_w = true;

        // External metrics (add more as you implement them)
        // bool adjusted_rand_index = true;
        // bool normalized_mutual_info = true;
    } metrics;
};

// ============================================================================
// Synthetic Data Generation
// ============================================================================
/**
 * @brief Generates synthetic clustered data using Gaussian blobs.
 *
 * Creates n_clusters Gaussian distributions with random centers and
 * specified standard deviation.
 */
Eigen::MatrixXd generate_blobs(
    int n_samples,
    int n_features,
    int n_clusters,
    double cluster_std,
    int random_seed
) {
    std::mt19937 gen(random_seed);
    std::uniform_real_distribution<> center_dist(-10.0, 10.0);
    std::normal_distribution<> noise_dist(0.0, cluster_std);

    // Generate random cluster centers
    Eigen::MatrixXd centers(n_clusters, n_features);
    for (int i = 0; i < n_clusters; ++i) {
        for (int j = 0; j < n_features; ++j) {
            centers(i, j) = center_dist(gen);
        }
    }

    // Generate samples
    Eigen::MatrixXd data(n_samples, n_features);
    int samples_per_cluster = n_samples / n_clusters;

    for (int c = 0; c < n_clusters; ++c) {
        int start_idx = c * samples_per_cluster;
        int end_idx = (c == n_clusters - 1) ? n_samples : (c + 1) * samples_per_cluster;

        for (int i = start_idx; i < end_idx; ++i) {
            for (int j = 0; j < n_features; ++j) {
                data(i, j) = centers(c, j) + noise_dist(gen);
            }
        }
    }

    return data;
}

// ============================================================================
// K-Means Algorithm
// ============================================================================
/**
 * @brief Simple K-means clustering implementation.
 *
 * @param data Input data matrix (n_samples, n_features)
 * @param k Number of clusters
 * @param max_iter Maximum iterations
 * @param tol Convergence tolerance
 * @param random_seed Random seed for initialization
 * @return Cluster labels for each sample
 */
Eigen::VectorXi kmeans(
    const Eigen::MatrixXd& data,
    int k,
    int max_iter,
    double tol,
    int random_seed
) {
    const int n_samples = data.rows();
    const int n_features = data.cols();

    // Initialize labels randomly
    std::mt19937 gen(random_seed);
    std::uniform_int_distribution<> label_dist(0, k - 1);

    Eigen::VectorXi labels(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        labels(i) = label_dist(gen);
    }

    Eigen::MatrixXd centers = Eigen::MatrixXd::Zero(k, n_features);
    Eigen::MatrixXd old_centers = Eigen::MatrixXd::Zero(k, n_features);

    // K-means iterations
    for (int iter = 0; iter < max_iter; ++iter) {
        // Save old centers
        old_centers = centers;

        // Update centers
        centers.setZero();
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);

        for (int i = 0; i < n_samples; ++i) {
            int label = labels(i);
            centers.row(label) += data.row(i);
            counts(label)++;
        }

        for (int c = 0; c < k; ++c) {
            if (counts(c) > 0) {
                centers.row(c) /= counts(c);
            } else {
                // Reinitialize empty cluster
                centers.row(c) = data.row(label_dist(gen) % n_samples);
            }
        }

        // Assign points to nearest center
        for (int i = 0; i < n_samples; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_label = 0;

            for (int c = 0; c < k; ++c) {
                double dist = (data.row(i) - centers.row(c)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_label = c;
                }
            }

            labels(i) = best_label;
        }

        // Check convergence
        double center_shift = (centers - old_centers).norm();
        if (center_shift < tol) {
            break;
        }
    }

    return labels;
}

// ============================================================================
// Metrics Results Structure
// ============================================================================
struct MetricResult {
    std::string metric_name;
    std::vector<double> values;  // One value per k
    int best_k;
    double total_computation_time_ms;
    MetricDirection direction;
    MetricCategory category;
};

struct MetricsResults {
    std::vector<int> k_values;
    std::vector<MetricResult> metric_results;
};

// ============================================================================
// Initialize Metric Registry
// ============================================================================
/**
 * @brief Initializes the metric registry with all available metrics.
 *
 * TO ADD A NEW METRIC:
 * 1. Add the metric implementation to metrics.hpp
 * 2. Add an enable/disable flag to Config::MetricFlags
 * 3. Add a single line here following the pattern below
 */
MetricRegistry initialize_metrics(const Config& config) {
    MetricRegistry registry;

    // ========================================================================
    // INTERNAL METRICS
    // ========================================================================

    if (config.metrics.ball_hall) {
        registry.add_metric(MetricInfo(
            "Ball-Hall",
            [](const Eigen::Ref<const Eigen::MatrixXd>& data,
               const Eigen::Ref<const Eigen::VectorXi>& labels) {
                return qc::internal::Metrics::ball_hall_index<double>(data, labels);
            },
            MetricDirection::LOWER_IS_BETTER,
            MetricCategory::INTERNAL
        ));
    }

    if (config.metrics.calinski_harabasz) {
        registry.add_metric(MetricInfo(
            "Calinski-Harabasz",
            [](const Eigen::Ref<const Eigen::MatrixXd>& data,
               const Eigen::Ref<const Eigen::VectorXi>& labels) {
                return qc::internal::Metrics::calinski_harabasz_index<double>(data, labels);
            },
            MetricDirection::HIGHER_IS_BETTER,
            MetricCategory::INTERNAL
        ));
    }

    if (config.metrics.trace_w) {
        registry.add_metric(MetricInfo(
            "Trace(W)",
            [](const Eigen::Ref<const Eigen::MatrixXd>& data,
               const Eigen::Ref<const Eigen::VectorXi>& labels) {
                return qc::internal::Metrics::trace_w_index<double>(data, labels);
            },
            MetricDirection::LOWER_IS_BETTER,
            MetricCategory::INTERNAL
        ));
    }

    // ========================================================================
    // EXTERNAL METRICS (uncomment and add as you implement them)
    // ========================================================================

    // if (config.metrics.adjusted_rand_index) {
    //     registry.add_metric(MetricInfo(
    //         "Adjusted Rand Index",
    //         [](const Eigen::Ref<const Eigen::MatrixXd>& data,
    //            const Eigen::Ref<const Eigen::VectorXi>& labels) {
    //             return qc::external::Metrics::adjusted_rand_index<double>(data, labels);
    //         },
    //         MetricDirection::HIGHER_IS_BETTER,
    //         MetricCategory::EXTERNAL
    //     ));
    // }

    // Sort metrics alphabetically
    registry.sort_alphabetically();

    return registry;
}

// ============================================================================
// Compute Metrics for All k Values
// ============================================================================
MetricsResults compute_metrics_for_all_k(
    const Eigen::MatrixXd& data,
    const Config& config,
    MetricRegistry& registry
) {
    MetricsResults results;
    results.k_values.reserve(config.MAX_CLUSTERS - config.MIN_CLUSTERS + 1);

    std::cout << "\n" << fmt::format("Running K-means for k = {} to {}...\n",
                                      config.MIN_CLUSTERS, config.MAX_CLUSTERS);

    // Store labels for each k value
    std::vector<Eigen::VectorXi> all_labels;
    all_labels.reserve(config.MAX_CLUSTERS - config.MIN_CLUSTERS + 1);

    // Run K-means for each k value and store labels
    for (int k = config.MIN_CLUSTERS; k <= config.MAX_CLUSTERS; ++k) {
        std::cout << fmt::format("  k = {}: Running K-means... ", k) << std::flush;

        Eigen::VectorXi labels = kmeans(data, k, config.MAX_KMEANS_ITER,
                                       config.KMEANS_TOL, config.RANDOM_SEED);
        all_labels.push_back(labels);
        results.k_values.push_back(k);

        std::cout << "Done\n";
    }

    // Compute metrics
    std::cout << "\nComputing metrics...\n";

    for (auto& metric_info : registry.get_metrics()) {
        if (!metric_info.enabled) continue;

        std::cout << fmt::format("  {}: ", metric_info.name) << std::flush;

        MetricResult result;
        result.metric_name = metric_info.name;
        result.direction = metric_info.direction;
        result.category = metric_info.category;
        result.values.reserve(all_labels.size());

        // Time the metric computation across all k values
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < all_labels.size(); ++i) {
            double value = metric_info.function(data, all_labels[i]);
            result.values.push_back(value);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.total_computation_time_ms = duration.count() / 1000.0;

        // Find best k
        if (metric_info.direction == MetricDirection::LOWER_IS_BETTER) {
            auto min_it = std::min_element(result.values.begin(), result.values.end());
            result.best_k = results.k_values[std::distance(result.values.begin(), min_it)];
        } else {
            auto max_it = std::max_element(result.values.begin(), result.values.end());
            result.best_k = results.k_values[std::distance(result.values.begin(), max_it)];
        }

        results.metric_results.push_back(result);

        std::cout << fmt::format("Done ({:.2f} ms)\n", result.total_computation_time_ms);
    }

    return results;
}

// ============================================================================
// Create Configuration Summary Table
// ============================================================================
Table create_config_table(const Config& config) {
    Table table;

    // Title row - must match column count
    table.add_row({"Configuration Parameters", ""});
    table[0].format()
        .font_align(FontAlign::center)
        .font_color(Color::cyan)
        .font_style({FontStyle::bold, FontStyle::underline});

    // Merge title row cells to span both columns
    table[0][0].format();

    // Parameter rows
    table.add_row({"Parameter", "Value"});
    table[1].format()
        .font_align(FontAlign::center)
        .font_color(Color::yellow)
        .font_style({FontStyle::bold});

    table.add_row({"MIN_CLUSTERS", std::to_string(config.MIN_CLUSTERS)});
    table.add_row({"MAX_CLUSTERS", std::to_string(config.MAX_CLUSTERS)});
    table.add_row({"N_SAMPLES", std::to_string(config.N_SAMPLES)});
    table.add_row({"N_FEATURES", std::to_string(config.N_FEATURES)});
    table.add_row({"CLUSTER_STD", fmt::format("{:.2f}", config.CLUSTER_STD)});
    table.add_row({"TRUE_CLUSTERS", std::to_string(config.TRUE_CLUSTERS)});
    table.add_row({"RANDOM_SEED", std::to_string(config.RANDOM_SEED)});
    table.add_row({"MAX_KMEANS_ITER", std::to_string(config.MAX_KMEANS_ITER)});
    table.add_row({"KMEANS_TOL", fmt::format("{:.0e}", config.KMEANS_TOL)});

    // Format data rows
    for (size_t i = 2; i < 11; ++i) {
        table[i][0].format().font_color(Color::cyan);
        table[i][1].format().font_align(FontAlign::right);
    }

    // Border styling
    table.format()
        .border_color(Color::cyan)
        .corner_color(Color::yellow);

    return table;
}

// ============================================================================
// Create Metrics Summary Table
// ============================================================================
Table create_metrics_table(const MetricsResults& results) {
    Table table;

    // Calculate total number of columns: Metric + k_values + Best k + Direction + Time
    const size_t num_columns = 1 + results.k_values.size() + 3;

    // Title row - create with correct number of columns
    Table::Row_t title_row;
    title_row.push_back("Internal Clustering Validation Metrics");
    for (size_t i = 1; i < num_columns; ++i) {
        title_row.push_back("");
    }
    table.add_row(title_row);
    table[0].format()
        .font_align(FontAlign::center)
        .font_color(Color::cyan)
        .font_style({FontStyle::bold, FontStyle::underline});

    // Merge title row to span all columns
    table[0][0].format();

    // Header row: Metric name + k values + Best k + Direction + Time
    Table::Row_t header_row = {"Metric"};
    for (int k : results.k_values) {
        header_row.push_back(fmt::format("k={}", k));
    }
    header_row.push_back("Best k");
    header_row.push_back("Direction");
    header_row.push_back("Time (ms)");
    table.add_row(header_row);

    table[1].format()
        .font_align(FontAlign::center)
        .font_color(Color::yellow)
        .font_style({FontStyle::bold});

    // Add a row for each metric
    size_t row_idx = 2;
    for (const auto& metric : results.metric_results) {
        Table::Row_t metric_row;

        // Add metric name with direction indicator
        std::string name_with_arrow = metric.metric_name;
        if (metric.direction == MetricDirection::LOWER_IS_BETTER) {
            name_with_arrow += " down";
        } else {
            name_with_arrow += " up";
        }
        metric_row.push_back(name_with_arrow);

        // Add metric values for each k
        for (size_t i = 0; i < metric.values.size(); ++i) {
            metric_row.push_back(fmt::format("{:.4f}", metric.values[i]));
        }

        // Add best k
        metric_row.push_back(std::to_string(metric.best_k));

        // Add direction text
        std::string direction_text = (metric.direction == MetricDirection::LOWER_IS_BETTER)
                                     ? "Lower" : "Higher";
        metric_row.push_back(direction_text);

        // Add computation time
        metric_row.push_back(fmt::format("{:.2f}", metric.total_computation_time_ms));

        table.add_row(metric_row);

        // Format metric name column
        table[row_idx][0].format()
            .font_color(Color::cyan)
            .font_style({FontStyle::bold});

        // Highlight best value for this metric
        for (size_t i = 0; i < metric.values.size(); ++i) {
            int k = results.k_values[i];
            if (k == metric.best_k) {
                table[row_idx][i + 1].format()
                    .font_color(Color::green)
                    .font_style({FontStyle::bold})
                    .font_background_color(Color::yellow);
            }
        }

        // Format "Best k" column
        table[row_idx][metric.values.size() + 1].format()
            .font_color(Color::green)
            .font_style({FontStyle::bold})
            .font_align(FontAlign::center);

        // Format "Direction" column
        table[row_idx][metric.values.size() + 2].format()
            .font_color(Color::magenta)
            .font_align(FontAlign::center);

        // Format "Time" column
        table[row_idx][metric.values.size() + 3].format()
            .font_align(FontAlign::right);

        row_idx++;
    }

    // Border styling
    table.format()
        .border_color(Color::cyan)
        .corner_color(Color::yellow);

    return table;
}

// ============================================================================
// Main Function
// ============================================================================
int main() {
    // Print banner
    std::cout << "\n";
    std::cout << "[================================================================]\n";
    std::cout << "|  Clustering Validation Metrics - K-means Testing Framework     |\n";
    std::cout << "[================================================================]\n";
    std::cout << "\n";

    // Initialize configuration
    Config config;

    // Initialize metric registry
    MetricRegistry registry = initialize_metrics(config);

    std::cout << fmt::format("Loaded {} metrics ({} enabled)\n",
                            registry.get_metrics().size(),
                            registry.enabled_count());

    // Display configuration
    const Table config_table = create_config_table(config);
    std::cout << "\n" << config_table << "\n";

    // Generate synthetic data
    std::cout << fmt::format("\nGenerating synthetic data with {} true clusters...\n",
                            config.TRUE_CLUSTERS);
    const Eigen::MatrixXd data = generate_blobs(
        config.N_SAMPLES,
        config.N_FEATURES,
        config.TRUE_CLUSTERS,
        config.CLUSTER_STD,
        config.RANDOM_SEED
    );
    std::cout << fmt::format(" Generated {} samples with {} features\n",
                            data.rows(), data.cols());

    // Compute metrics for all k values
    MetricsResults results = compute_metrics_for_all_k(data, config, registry);

    std::cout << "\n";
    std::cout << "[================================================================]\n";
    std::cout << "|                     Metrics Summary                            |\n";
    std::cout << "[================================================================]\n";
    std::cout << "\n";

    // Display metrics table
    const Table metrics_table = create_metrics_table(results);
    std::cout << metrics_table << "\n\n";

    // Summary
    std::cout << "Summary:\n";
    std::cout << fmt::format("  True number of clusters: {}\n", config.TRUE_CLUSTERS);
    std::cout << "  Metric recommendations:\n";

    for (const auto& metric : results.metric_results) {
        std::cout << fmt::format("    - {} suggests k = {}\n",
                                metric.metric_name, metric.best_k);
    }

    // Check for consensus
    if (!results.metric_results.empty()) {
        int first_best_k = results.metric_results[0].best_k;
        const bool all_agree = std::all_of(
            results.metric_results.begin(),
            results.metric_results.end(),
            [first_best_k](const MetricResult& r) { return r.best_k == first_best_k; }
        );

        if (all_agree) {
            std::cout << fmt::format("\n All metrics agree on k = {} (Ground truth: k = {})\n",
                                    first_best_k, config.TRUE_CLUSTERS);
        } else {
            std::cout << "\nMetrics disagree on optimal k value\n";

            // Count votes for each k
            std::map<int, int> k_votes;
            for (const auto& metric : results.metric_results) {
                k_votes[metric.best_k]++;
            }

            // Find majority
            const auto max_votes = std::max_element(
                k_votes.begin(), k_votes.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );

            if (max_votes != k_votes.end() && max_votes->second > 1) {
                std::cout << fmt::format("  Most metrics ({}/{}) suggest k = {}\n",
                                        max_votes->second,
                                        results.metric_results.size(),
                                        max_votes->first);
            }
        }
    }

    fmt::print("\n");

    return 0;
}