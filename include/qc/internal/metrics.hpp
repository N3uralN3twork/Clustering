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
class Metrics {
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
     * @brief Calculates the Calinski-Harabasz index (1974).
     *
     * Also known as the Variance Ratio Criterion (VRC).
     * Formula: (BGSS / (k - 1)) / (WGSS / (n - k))
     *
     * A higher CH score relates to a model with better-defined clusters.
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
     * @brief Calculates the Trace(W) index (part of Friedman-Rubin).
     *
     * This index is simply the trace of the within-cluster scatter matrix (W).
     * Lower values are generally better, indicating less scatter within clusters.
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
        auto centers = utils::get_centroids<T>(data, labels);
        if (centers.rows() == 0) {
            return T{0.0};
        }

        auto scatter = utils::get_scatter_matrices<T>(data, labels, centers);

        // The trace is the sum of the diagonal elements
        return scatter.WITHIN.trace();
    }

};

} // namespace qc::internal
