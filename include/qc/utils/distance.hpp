#pragma once

#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <execution>
#include <algorithm>
#include <optional>

namespace qc {

// ============================================================================
// Distance Metric Concepts and Types
// ============================================================================

/// Concept for valid scalar types
template<typename T>
concept Scalar = std::is_floating_point_v<T>;

/// Distance metric function signature
template<typename Scalar>
using DistanceMetric = Scalar(*)(
    const Eigen::Ref<const Eigen::VectorX<Scalar>>&,
    const Eigen::Ref<const Eigen::VectorX<Scalar>>&
);

/// Enum for built-in distance metrics
enum class Metric {
    Euclidean,
    SquaredEuclidean,
    Manhattan,
    Chebyshev,
    Minkowski,
    Cosine,
    Correlation,
    Hamming
};

// ============================================================================
// Individual Distance Metrics
// ============================================================================

namespace metrics {

/// Euclidean distance: sqrt(sum((x - y)^2))
template<Scalar T>
inline T euclidean(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    return (x - y).norm();
}

/// Squared Euclidean distance: sum((x - y)^2)
/// Faster than euclidean; use when only comparisons needed
template<Scalar T>
inline T squared_euclidean(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    return (x - y).squaredNorm();
}

/// Manhattan distance: sum(|x - y|)
template<Scalar T>
inline T manhattan(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    return (x - y).lpNorm<1>();
}

/// Chebyshev distance: max(|x - y|)
template<Scalar T>
inline T chebyshev(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    return (x - y).lpNorm<Eigen::Infinity>();
}

/// Minkowski distance: (sum(|x - y|^p))^(1/p)
template<Scalar T>
inline T minkowski(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y,
    T p
) {
    return std::pow((x - y).array().abs().pow(p).sum(), 1.0 / p);
}

/// Cosine distance: 1 - (x·y) / (||x|| * ||y||)
template<Scalar T>
inline T cosine(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    T dot = x.dot(y);
    T norm_product = x.norm() * y.norm();
    if (norm_product < std::numeric_limits<T>::epsilon()) {
        return T{0};
    }
    return T{1} - (dot / norm_product);
}

/// Correlation distance: 1 - correlation(x, y)
template<Scalar T>
inline T correlation(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    Eigen::VectorX<T> x_centered = x.array() - x.mean();
    Eigen::VectorX<T> y_centered = y.array() - y.mean();
    return cosine(x_centered, y_centered);
}

/// Hamming distance: proportion of differing elements
template<Scalar T>
inline T hamming(
    const Eigen::Ref<const Eigen::VectorX<T>>& x,
    const Eigen::Ref<const Eigen::VectorX<T>>& y
) {
    return (x.array() != y.array()).template cast<T>().mean();
}

} // namespace metrics

// ============================================================================
// Pairwise Distance Computation
// ============================================================================

/// Options for distance computation
struct DistanceOptions {
    bool parallel = true;           // Use parallel execution
    bool upper_triangular_only = true;  // Only compute upper triangle (symmetric)
    bool include_diagonal = false;  // Include diagonal (usually 0)
    size_t cache_line_size = 64;   // For cache-friendly access patterns
};

/// Compute pairwise distances between all points
template<Scalar T>
class PairwiseDistances {
public:
    using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXT = Eigen::VectorX<T>;

    /// Compute full distance matrix
    static MatrixXT compute(
        const Eigen::Ref<const MatrixXT>& X,
        Metric metric = Metric::Euclidean,
        const DistanceOptions& opts = {}
    ) {
        const auto n_samples = X.rows();
        MatrixXT distances = MatrixXT::Zero(n_samples, n_samples);

        auto dist_func = get_metric_function(metric);

        if (opts.parallel) {
            compute_parallel(X, distances, dist_func, opts);
        } else {
            compute_sequential(X, distances, dist_func, opts);
        }

        // Mirror if we only computed upper triangle
        if (opts.upper_triangular_only) {
            distances.triangularView<Eigen::Lower>() =
                distances.triangularView<Eigen::Upper>().transpose();
        }

        return distances;
    }

    /// Compute condensed distance vector (upper triangle only, no diagonal)
    /// Returns vector of size n*(n-1)/2
    static std::vector<T> compute_condensed(
        const Eigen::Ref<const MatrixXT>& X,
        Metric metric = Metric::Euclidean
    ) {
        const auto n_samples = X.rows();
        const size_t n_distances = (n_samples * (n_samples - 1)) / 2;
        std::vector<T> distances(n_distances);

        auto dist_func = get_metric_function(metric);

        size_t idx = 0;
        for (Eigen::Index i = 0; i < n_samples - 1; ++i) {
            for (Eigen::Index j = i + 1; j < n_samples; ++j) {
                distances[idx++] = dist_func(X.row(i), X.row(j));
            }
        }

        return distances;
    }

    /// Convert condensed form to square matrix
    static MatrixXT condensed_to_square(
        const std::vector<T>& condensed,
        size_t n_samples
    ) {
        MatrixXT distances = MatrixXT::Zero(n_samples, n_samples);

        size_t idx = 0;
        for (size_t i = 0; i < n_samples - 1; ++i) {
            for (size_t j = i + 1; j < n_samples; ++j) {
                distances(i, j) = condensed[idx];
                distances(j, i) = condensed[idx];
                ++idx;
            }
        }

        return distances;
    }

private:
    static DistanceMetric<T> get_metric_function(Metric metric) {
        switch (metric) {
            case Metric::Euclidean:
                return metrics::euclidean<T>;
            case Metric::SquaredEuclidean:
                return metrics::squared_euclidean<T>;
            case Metric::Manhattan:
                return metrics::manhattan<T>;
            case Metric::Chebyshev:
                return metrics::chebyshev<T>;
            case Metric::Cosine:
                return metrics::cosine<T>;
            case Metric::Correlation:
                return metrics::correlation<T>;
            case Metric::Hamming:
                return metrics::hamming<T>;
            default:
                return metrics::euclidean<T>;
        }
    }

    static void compute_sequential(
        const Eigen::Ref<const MatrixXT>& X,
        Eigen::Ref<MatrixXT> distances,
        DistanceMetric<T> dist_func,
        const DistanceOptions& opts
    ) {
        const auto n_samples = X.rows();

        for (Eigen::Index i = 0; i < n_samples; ++i) {
            Eigen::Index j_start = opts.upper_triangular_only ? i + 1 : 0;

            if (!opts.upper_triangular_only && opts.include_diagonal) {
                distances(i, i) = T{0};
            }

            for (Eigen::Index j = j_start; j < n_samples; ++j) {
                if (i == j && !opts.include_diagonal) continue;
                distances(i, j) = dist_func(X.row(i), X.row(j));
            }
        }
    }

    static void compute_parallel(
        const Eigen::Ref<const MatrixXT>& X,
        Eigen::Ref<MatrixXT> distances,
        DistanceMetric<T> dist_func,
        const DistanceOptions& opts
    ) {
        const auto n_samples = X.rows();
        std::vector<Eigen::Index> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        std::for_each(
            std::execution::par_unseq,
            indices.begin(), indices.end(),
            [&](Eigen::Index i) {
                Eigen::Index j_start = opts.upper_triangular_only ? i + 1 : 0;

                for (Eigen::Index j = j_start; j < n_samples; ++j) {
                    if (i == j && !opts.include_diagonal) continue;
                    distances(i, j) = dist_func(X.row(i), X.row(j));
                }
            }
        );
    }
};

// ============================================================================
// Centroid Distance Computations (optimized for clustering)
// ============================================================================

/// Compute distances from points to centroids
template<Scalar T>
class CentroidDistances {
public:
    using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXT = Eigen::VectorX<T>;

    /// Compute distance from each point to each centroid
    /// Returns: (n_samples × n_centroids) matrix
    static MatrixXT compute(
        const Eigen::Ref<const MatrixXT>& X,
        const Eigen::Ref<const MatrixXT>& centroids,
        Metric metric = Metric::Euclidean,
        bool parallel = true
    ) {
        const auto n_samples = X.rows();
        const auto n_centroids = centroids.rows();
        MatrixXT distances(n_samples, n_centroids);

        auto dist_func = PairwiseDistances<T>::get_metric_function(metric);

        if (parallel) {
            std::vector<Eigen::Index> sample_indices(n_samples);
            std::iota(sample_indices.begin(), sample_indices.end(), 0);

            std::for_each(
                std::execution::par_unseq,
                sample_indices.begin(), sample_indices.end(),
                [&](Eigen::Index i) {
                    for (Eigen::Index k = 0; k < n_centroids; ++k) {
                        distances(i, k) = dist_func(X.row(i), centroids.row(k));
                    }
                }
            );
        } else {
            for (Eigen::Index i = 0; i < n_samples; ++i) {
                for (Eigen::Index k = 0; k < n_centroids; ++k) {
                    distances(i, k) = dist_func(X.row(i), centroids.row(k));
                }
            }
        }

        return distances;
    }

    /// Compute distance from each point to its assigned centroid only
    /// Returns: vector of length n_samples
    static VectorXT compute_assigned(
        const Eigen::Ref<const MatrixXT>& X,
        const Eigen::Ref<const MatrixXT>& centroids,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        Metric metric = Metric::Euclidean
    ) {
        const auto n_samples = X.rows();
        VectorXT distances(n_samples);

        auto dist_func = PairwiseDistances<T>::get_metric_function(metric);

        for (Eigen::Index i = 0; i < n_samples; ++i) {
            const int cluster_id = labels(i);
            distances(i) = dist_func(X.row(i), centroids.row(cluster_id));
        }

        return distances;
    }

    /// Find minimum distance and corresponding centroid for each point
    /// Returns: pair of (distances, nearest_centroid_indices)
    static std::pair<VectorXT, Eigen::VectorXi> compute_nearest(
        const Eigen::Ref<const MatrixXT>& X,
        const Eigen::Ref<const MatrixXT>& centroids,
        Metric metric = Metric::Euclidean,
        bool parallel = true
    ) {
        MatrixXT all_distances = compute(X, centroids, metric, parallel);

        const auto n_samples = X.rows();
        VectorXT min_distances(n_samples);
        Eigen::VectorXi nearest_indices(n_samples);

        for (Eigen::Index i = 0; i < n_samples; ++i) {
            Eigen::Index min_idx;
            min_distances(i) = all_distances.row(i).minCoeff(&min_idx);
            nearest_indices(i) = static_cast<int>(min_idx);
        }

        return {min_distances, nearest_indices};
    }
};

// ============================================================================
// Distance Matrix Cache (for reuse across multiple metrics)
// ============================================================================

template<Scalar T>
class DistanceMatrixCache {
public:
    using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    DistanceMatrixCache() = default;

    /// Get or compute distance matrix
    const MatrixXT& get_or_compute(
        const Eigen::Ref<const MatrixXT>& X,
        Metric metric = Metric::Euclidean,
        const DistanceOptions& opts = {}
    ) {
        // Simple cache key based on data pointer and metric
        auto key = std::make_pair(X.data(), static_cast<int>(metric));

        if (cache_key_ != key || !cached_distances_.has_value()) {
            cached_distances_ = PairwiseDistances<T>::compute(X, metric, opts);
            cache_key_ = key;
        }

        return *cached_distances_;
    }

    /// Get condensed form or compute
    const std::vector<T>& get_or_compute_condensed(
        const Eigen::Ref<const MatrixXT>& X,
        Metric metric = Metric::Euclidean
    ) {
        auto key = std::make_pair(X.data(), static_cast<int>(metric));

        if (condensed_cache_key_ != key || !cached_condensed_.has_value()) {
            cached_condensed_ = PairwiseDistances<T>::compute_condensed(X, metric);
            condensed_cache_key_ = key;
        }

        return *cached_condensed_;
    }

    void clear() {
        cached_distances_.reset();
        cached_condensed_.reset();
    }

private:
    std::optional<MatrixXT> cached_distances_;
    std::optional<std::vector<T>> cached_condensed_;
    std::pair<const T*, int> cache_key_{nullptr, -1};
    std::pair<const T*, int> condensed_cache_key_{nullptr, -1};
};

// ============================================================================
// Specialized Distance Computations
// ============================================================================

/// Compute minimum inter-cluster distance (for Dunn index, etc.)
template<Scalar T>
T min_intercluster_distance(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& centroids,
    Metric metric = Metric::Euclidean
) {
    const auto n_clusters = centroids.rows();
    if (n_clusters < 2) return T{0};

    auto dist_func = PairwiseDistances<T>::get_metric_function(metric);
    T min_dist = std::numeric_limits<T>::max();

    for (Eigen::Index i = 0; i < n_clusters - 1; ++i) {
        for (Eigen::Index j = i + 1; j < n_clusters; ++j) {
            T dist = dist_func(centroids.row(i), centroids.row(j));
            min_dist = std::min(min_dist, dist);
        }
    }

    return min_dist;
}

/// Compute maximum intra-cluster distance (for Dunn index, etc.)
template<Scalar T>
T max_intracluster_distance(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& X,
    const Eigen::Ref<const Eigen::VectorXi>& labels,
    int cluster_id,
    Metric metric = Metric::Euclidean
) {
    // Extract points belonging to this cluster
    std::vector<Eigen::Index> cluster_indices;
    for (Eigen::Index i = 0; i < labels.size(); ++i) {
        if (labels(i) == cluster_id) {
            cluster_indices.push_back(i);
        }
    }

    if (cluster_indices.size() < 2) return T{0};

    auto dist_func = PairwiseDistances<T>::get_metric_function(metric);
    T max_dist = T{0};

    for (size_t i = 0; i < cluster_indices.size() - 1; ++i) {
        for (size_t j = i + 1; j < cluster_indices.size(); ++j) {
            T dist = dist_func(
                X.row(cluster_indices[i]),
                X.row(cluster_indices[j])
            );
            max_dist = std::max(max_dist, dist);
        }
    }

    return max_dist;
}

} // namespace qc