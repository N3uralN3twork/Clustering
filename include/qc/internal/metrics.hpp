#pragma once
// Import helper functions from your other files
#include "qc/utils/utils.hpp"
#include "qc/utils/distance.hpp" // For qc::Scalar concept
#include <limits> // For std::numeric_limits

namespace qc::internal {

/**
 * @brief Provides internal cluster validation metrics.
 *
 * This class follows the pattern from utils.hpp, offering static methods
 * for calculating various clustering indices.
 *
 * (Note: Renamed from 'Utils' in the stub to 'Metrics' to avoid
 * confusion with Utils.hpp and to match the Python class name).
 */
class Metrics
{
public:
    /**
     * @brief Calculates the Ball-Hall index (1995).
     *
     * The Ball-Hall Index is the mean of the within-cluster dispersion,
     * calculated as: Total WGSS / k
     * where:
     * - Total WGSS is the total Within-Cluster Sum of Squares.
     * - k is the number of clusters.
     *
     * <i>Criterion:</i> <b>Largest difference</b>
     *
     * This implementation is highly efficient as it reuses the Utils::gss
     * function, which computes total WGSS in a single pass over the data.
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return T The calculated Ball-Hall index.
     */
    template<qc::Scalar T>
    static T ball_hall_index(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    ) {

        // --- Input Validation (Lightweight) ---
        if (data.rows() == 0 || labels.size() == 0) {
            return T{0.0};
        }

        // --- Step 1: Calculate GSS (WGSS, BGSS, TSS, Centers) ---
        // This reuses your highly optimized gss function from utils.hpp.
        // We must explicitly provide the template argument <T> here.
        auto gss_result = utils::gss(data, labels);

        // --- Step 2: Get k (number of clusters) ---
        // We get 'k' from the number of rows in the centers matrix.
        const long k = gss_result.centers.rows();

        // --- Step 3: Handle Division by Zero ---
        if (k == 0) {
            return T{0.0};
        }

        // --- Step 4: Compute Ball-Hall Index ---
        // The index is simply Total WGSS / k.
        return gss_result.wgss / static_cast<T>(k);
    }

    /**
    * @brief Calculates the C index by Hubert & Levin (1976).
    *
    * <a href="https://github.com/johnvorsten/py_cindex">John Vorsten</a>
    *
    * <i>Criterion:</i> <b>Minimum</b>
    *
    * @tparam T A floating-point type (e.g., double, float).
    * @param data The input data matrix (n_samples, n_features).
    * @param labels A vector of cluster labels (n_samples, 1).
    * @return T The calculated Ball-Hall index.
    */
    template<qc::Scalar T>
    static T c_index(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    )
    {
        // --- Input Validation ---
        const long n = data.rows();
        if (n < 2 || labels.size() < 2) {
            return T{0.0};
        }

        // --- Step 1: Compute condensed distance vector (more memory efficient) ---
        std::vector<T> distances = qc::PairwiseDistances<T>::compute_condensed(
            data,
            qc::Metric::Euclidean
        );

        if (distances.empty()) {
            return T{0.0};
        }

        // --- Step 2: Calculate N_w (number of within-cluster pairs) ---
        // N_w = sum over all clusters of (n_i * (n_i - 1) / 2)
        const int k = labels.maxCoeff() + 1;
        Eigen::VectorXi cluster_counts = Eigen::VectorXi::Zero(k);

        for (long i = 0; i < n; ++i) {
            const int cluster_id = labels(i);
            if (cluster_id >= 0 && cluster_id < k) {
                ++cluster_counts(cluster_id);
            }
        }

        long N_w = 0;
        for (int c = 0; c < k; ++c) {
            const long n_c = cluster_counts(c);
            if (n_c > 1) {
                N_w += (n_c * (n_c - 1)) / 2;
            }
        }

        if (N_w == 0 || N_w > static_cast<long>(distances.size())) {
            return T{0.0};
        }

        // --- Step 3: Calculate S_w (sum of within-cluster distances) ---
        // Iterate through condensed distance vector and sum distances for same-cluster pairs
        T S_w = T{0.0};
        size_t idx = 0;

        for (long i = 0; i < n - 1; ++i) {
            const int label_i = labels(i);

            for (long j = i + 1; j < n; ++j, ++idx) {
                const int label_j = labels(j);

                // If both points are in the same cluster, add to S_w
                if (label_i == label_j) {
                    S_w += distances[idx];
                }
            }
        }

        // --- Step 4: Find S_min and S_max efficiently ---
        // Use partial_sort to find the N_w smallest and largest distances
        // This is O(n log N_w) instead of O(n log n) for full sort

        // For S_min: find N_w smallest values
        std::vector<T> distances_copy = distances; // Need a copy for partial_sort
        std::partial_sort(
            distances_copy.begin(),
            distances_copy.begin() + N_w,
            distances_copy.end()
        );

        T S_min = T{0.0};
        for (long i = 0; i < N_w; ++i) {
            S_min += distances_copy[i];
        }

        // For S_max: find N_w largest values
        // Use nth_element + partial_sort for efficiency
        std::nth_element(
            distances.begin(),
            distances.end() - N_w,
            distances.end()
        );
        std::partial_sort(
            distances.end() - N_w,
            distances.end(),
            distances.end()
        );

        T S_max = T{0.0};
        for (auto it = distances.end() - N_w; it != distances.end(); ++it) {
            S_max += *it;
        }

        // --- Step 5: Calculate C-Index ---
        const T denominator = S_max - S_min;
        if (std::abs(denominator) < std::numeric_limits<T>::epsilon()) {
            return T{0.0}; // Avoid division by zero
        }

        return (S_w - S_min) / denominator;
    }

    /**
     * @brief Calculates the Calinski-Harabasz index (1974).
     *
     * Also known as the Variance Ratio Criterion (VRC).
     * Formula: (BGSS / (k - 1)) / (WGSS / (n - k))
     *
     * <i>Criterion:</i> <b>Maximum</b>
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return T The calculated Calinski-Harabasz index.
     */
    template<qc::Scalar T>
    static T calinski_harabasz_index(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    ) {
        const long n = data.rows();
        if (n == 0 || labels.size() == 0) {
            return T{0.0};
        }

        auto gss_result = utils::gss(data, labels);
        const long k = gss_result.centers.rows();

        // Handle edge cases
        if (k <= 1 || k >= n) {
            return T{0.0};
        }

        const T bgss_term = gss_result.bgss / static_cast<T>(k - 1);
        const T wgss_term = gss_result.wgss / static_cast<T>(n - k);

        if (wgss_term < std::numeric_limits<T>::epsilon()) {
            return T{0.0}; // Avoid division by zero
        }

        return bgss_term / wgss_term;
    }
    /**
         * @brief Calculates the Davies-Bouldin index (1979).
         *
         * The Davies-Bouldin Index measures the average similarity between each cluster
         * and its most similar cluster. Lower values indicate better clustering.
         *
         * Formula: DB = (1/k) * Σ max_j≠i[(S_i + S_j) / d(c_i, c_j)]
         * where:
         * - S_i is the average distance from points in cluster i to its centroid
         * - d(c_i, c_j) is the distance between centroids i and j
         * - k is the number of clusters
         *
         * <i>Criterion:</i> <b>Minimum</b>
         *
         * @tparam T A floating-point type (e.g., double, float).
         * @param data The input data matrix (n_samples, n_features).
         * @param labels A vector of cluster labels (n_samples, 1).
         * @return T The calculated Davies-Bouldin index.
         *
         * @note This implementation is optimized to:
         * - Reuse existing centroid calculations
         * - Compute intra-cluster distances in a single pass
         * - Use efficient distance computations
         * - Avoid unnecessary memory allocations
         *
         * References:
         * Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
         * IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2), 224-227.
    */
    template<qc::Scalar T>
    static T davies_bouldin_index(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    ) {
        // --- Input Validation ---
        const long n = data.rows();
        if (n < 2 || labels.size() < 2) {
            return T{0.0};
        }

        // --- Step 1: Calculate centroids (reuse existing function) ---
        auto centroids = utils::get_centroids<T>(data, labels);
        const int k = static_cast<int>(centroids.rows());

        if (k < 2) {
            return T{0.0}; // Need at least 2 clusters
        }

        // --- Step 2: Calculate intra-cluster distances (S_i) ---
        // S_i = average distance from points in cluster i to centroid i
        Eigen::Vector<T, Eigen::Dynamic> intra_dists = Eigen::Vector<T, Eigen::Dynamic>::Zero(k);
        Eigen::VectorXi cluster_counts = Eigen::VectorXi::Zero(k);

        // Single pass: accumulate distances and count points
        for (long i = 0; i < n; ++i) {
            const int cluster_id = labels(i);
            if (cluster_id < 0 || cluster_id >= k) continue;

            const T dist = (data.row(i) - centroids.row(cluster_id)).norm();
            intra_dists(cluster_id) += dist;
            ++cluster_counts(cluster_id);
        }

        // Calculate averages
        for (int i = 0; i < k; ++i) {
            if (cluster_counts(i) > 0) {
                intra_dists(i) /= static_cast<T>(cluster_counts(i));
            }
        }

        // Check for degenerate case
        if (intra_dists.isZero(std::numeric_limits<T>::epsilon())) {
            return T{0.0};
        }

        // --- Step 3: Calculate centroid distances ---
        // Use efficient pairwise distance computation
        auto centroid_distances = qc::PairwiseDistances<T>::compute(
            centroids,
            qc::Metric::Euclidean,
            qc::DistanceOptions{.parallel = false, .upper_triangular_only = false, .include_diagonal = true}
        );

        // Check for degenerate case
        bool all_zero = true;
        for (int i = 0; i < k && all_zero; ++i) {
            for (int j = 0; j < k; ++j) {
                if (i != j && centroid_distances(i, j) > std::numeric_limits<T>::epsilon()) {
                    all_zero = false;
                    break;
                }
            }
        }
        if (all_zero) {
            return T{0.0};
        }

        // --- Step 4: Calculate Davies-Bouldin Index ---
        // For each cluster i, find max_j≠i[(S_i + S_j) / d(c_i, c_j)]
        T db_sum = T{0.0};

        for (int i = 0; i < k; ++i) {
            T max_ratio = T{0.0};

            for (int j = 0; j < k; ++j) {
                if (i == j) continue;

                const T dist_ij = centroid_distances(i, j);
                if (dist_ij < std::numeric_limits<T>::epsilon()) {
                    continue; // Skip if centroids are identical
                }

                const T ratio = (intra_dists(i) + intra_dists(j)) / dist_ij;
                max_ratio = std::max(max_ratio, ratio);
            }

            db_sum += max_ratio;
        }

        return db_sum / static_cast<T>(k);
    }

    /**
    * @brief Calculates the Davies-Bouldin score.
     *
    * The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.
     *
     * <i>Criterion:</i> <b>Minimum</b>
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return T The calculated Trace(W) index.
     */


    /**
     * @brief Calculates the Trace(W) index (part of Friedman-Rubin).
     *
     * This index is simply the trace of the within-cluster scatter matrix (W).
     *
     * <i>Criterion:</i> <b>Maximum Difference</b>
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return T The calculated Trace(W) index.
     */
    template<qc::Scalar T>
    static T trace_w_index(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    ) {
        if (data.rows() == 0 || labels.size() == 0) {
            return T{0.0};
        }

        // Need centers to calculate scatter matrices
        auto centroids = utils::get_centroids<T>(data, labels);
        if (centroids.rows() == 0) {
            return T{0.0};
        }

        auto scatter = utils::get_scatter_matrices<T>(data, labels, centroids);

        // The trace is the sum of the diagonal elements
        return scatter.WITHIN.trace();
    }

};

} // namespace qc::internal
