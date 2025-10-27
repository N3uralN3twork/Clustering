#pragma once

#include <iostream> // For error reporting
#include "distance.hpp"

/**
 * @brief A collection of utility functions for clustering algorithms.
 *
 * This namespace provides fundamental "building block" functions
 * used by multiple validation indices.
 */
namespace qc::utils {

    /**
     * @brief A struct to hold the results of the gss() (Get Sum of Squares) calculation.
     */
    template<qc::Scalar T>
    struct GssResult {
        T wgss = T{0.0}; // Within-cluster Sum of Squares
        T bgss = T{0.0}; // Between-cluster Sum of Squares
        T tss = T{0.0};  // Total Sum of Squares (allmeandist in Python)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centers; // The calculated centroids
    };

    /**
     * @brief Calculates the centroid (mean) for each cluster.
     * *
     * This function performs a single pass over the dataset to efficiently
     * compute the mean of all points belonging to each cluster.
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return Eigen::Matrix<T, ...> A matrix where each row is the centroid (k, n_features).
     */
    template<qc::Scalar T>
    [[nodiscard]] Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> get_centroids(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    )  {

        // --- Input Validation ---
        if (data.rows() != labels.size()) {
            std::cerr << "Error in get_centroids: Data rows (" << data.rows()
                      << ") does not match labels size (" << labels.size() << ")." << std::endl;
            return {};
        }

        if (data.rows() == 0) {
            std::cerr << "Error in get_centroids: Input data is empty." << std::endl;
            return {};
        }

        // --- Initialization ---
        const int num_clusters = labels.maxCoeff() + 1;
        const long n_cols = data.cols();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centers =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(num_clusters, n_cols);
        Eigen::VectorXi cluster_counts = Eigen::VectorXi::Zero(num_clusters);

        // --- Calculation (Single Pass) ---
        for (long i = 0; i < data.rows(); ++i) {
            const int cluster_id = labels(i);

            // Add check for out-of-bounds labels
            if (cluster_id < 0 || cluster_id >= num_clusters) {
                 std::cerr << "Error in get_centroids: Label " << cluster_id
                           << " is out of range [0, " << num_clusters - 1 << "]." << std::endl;
                 continue; // Skip this invalid label
            }

            centers.row(cluster_id) += data.row(i);
            cluster_counts(cluster_id)++;
        }

        // --- Calculate Mean ---
        for (int i = 0; i < num_clusters; ++i) {
            if (cluster_counts(i) > 0) {
                centers.row(i) /= static_cast<T>(cluster_counts(i));
            }
            // If cluster_counts(i) == 0, center remains zero vector.
        }

        return centers;
    }

    /**
     * @brief Calculates the total, within-cluster, and between-cluster sum of squares.
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return GssResult<T> A struct containing wgss, bgss, tss, and the cluster centers.
     */
    template<qc::Scalar T>
    [[nodiscard]] GssResult<T> gss(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels
    )  {

        // --- Step 1: Calculate Centroids ---
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centers = get_centroids<T>(data, labels);

        if (centers.rows() == 0) {
            // get_centroids failed, return empty result
            return {};
        }

        // --- Step 2: Calculate Total Sum of Squares (TSS) ---
        const Eigen::RowVector<T, Eigen::Dynamic> allmean = data.colwise().mean();
        const T tss = (data.rowwise() - allmean).squaredNorm();

        // --- Step 3: Calculate Within-cluster Sum of Squares (WGSS) ---
        T wgss = T{0.0};
        const int num_clusters = static_cast<int>(centers.rows());
        for (long i = 0; i < data.rows(); ++i) {
            const int cluster_id = labels(i);

            // Add check for out-of-bounds labels
            if (cluster_id < 0 || cluster_id >= num_clusters) {
                 continue; // Skip this invalid label
            }

            wgss += (data.row(i) - centers.row(cluster_id)).squaredNorm();
        }

        // --- Step 4: Calculate Between-cluster Sum of Squares (BGSS) ---
        const T bgss = tss - wgss;

        return {wgss, bgss, tss, centers};
    }

    /**
     * @brief Calculates the Within-Cluster Sum of Squares (WGSS) for each cluster.
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @param centers The pre-calculated centroids (k, n_features).
     * @return Eigen::Vector<T, ...> A vector of size k, where each element i
     * is the WGSS for cluster i.
     */
    template<qc::Scalar T>
    [[nodiscard]] Eigen::Vector<T, Eigen::Dynamic> get_per_cluster_wgss(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& centers
    )  {
        const int k = static_cast<int>(centers.rows());
        Eigen::Vector<T, Eigen::Dynamic> per_cluster_wgss =
            Eigen::Vector<T, Eigen::Dynamic>::Zero(k);

        for (long i = 0; i < data.rows(); ++i) {
            const int cluster_id = labels(i);

            // Add check for out-of-bounds labels
            if (cluster_id < 0 || cluster_id >= k) {
                 continue; // Skip this invalid label
            }

            per_cluster_wgss(cluster_id) += (data.row(i) - centers.row(cluster_id)).squaredNorm();
        }

        return per_cluster_wgss;
    }

    /**
     * @brief Struct to hold the W, B, and T scatter matrices.
     * W = Within-cluster scatter matrix
     * B = Between-cluster scatter matrix
     * T = Total scatter matrix (T = W + B)
     */
    template<qc::Scalar T>
    struct ScatterMatrices {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> WITHIN;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> BETWEEN;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> TOTAL;
    };

    /**
     * @brief Calculates the within-cluster, between-cluster, and total scatter matrices.
     *
     * @tparam T A floating-point type (e.g., double, float).
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @param centers The pre-calculated centroids (k, n_features).
     * @return ScatterMatrices<T> A struct containing the W, B, and T matrices (all p x p).
     */
    template<qc::Scalar T>
    [[nodiscard]] ScatterMatrices<T> get_scatter_matrices(
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& centers
    )  {
        const long n = data.rows();
        const long p = data.cols();
        const int k = static_cast<int>(centers.rows());

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> WITHIN =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(p, p);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> BETWEEN =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(p, p);

        Eigen::VectorXi cluster_counts = Eigen::VectorXi::Zero(k);
        const Eigen::RowVector<T, Eigen::Dynamic> grand_mean = data.colwise().mean();

        // Calculate W (Within-cluster scatter matrix)
        for (long i = 0; i < n; ++i) {
            const int cluster_id = labels(i);

            // Add check for out-of-bounds labels
            if (cluster_id < 0 || cluster_id >= k) {
                 continue; // Skip this invalid label
            }

            const Eigen::RowVector<T, Eigen::Dynamic> diff_to_center = data.row(i) - centers.row(cluster_id);
            WITHIN += diff_to_center.transpose() * diff_to_center; // (p,1) * (1,p) -> (p,p)
            ++cluster_counts(cluster_id);
        }

        // Calculate B (Between-cluster scatter matrix)
        for (int j = 0; j < k; ++j) {
            if (cluster_counts(j) > 0) {
                const Eigen::RowVector<T, Eigen::Dynamic> diff_to_mean = centers.row(j) - grand_mean;
                BETWEEN += cluster_counts(j) * (diff_to_mean.transpose() * diff_to_mean);
            }
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Total = WITHIN +BETWEEN;

        return {WITHIN, BETWEEN, Total};
    }

} // namespace qc::utils
