#pragma once

#include <Eigen/Dense>
#include <iostream> // For error reporting

/**
 * @brief A collection of utility functions for clustering algorithms.
 *
 * This is a header-only class.
 */
class Utils {
public:
    /**
     * @brief Calculates the centroid (mean) for each cluster.
     * *
     * This function performs a single pass over the dataset to efficiently
     * compute the mean of all points belonging to each cluster.
     *
     * @param data The input data matrix where each row is a sample and each
     * column is a feature (n_samples, n_features).
     * @param labels A vector of cluster labels for each sample (n_samples, 1).
     * Labels must be 0-indexed (i.e., from 0 to k-1).
     * @return Eigen::MatrixXd A matrix where each row is the centroid for a
     * cluster (k, n_features), where k is the number of clusters.
     */
    static Eigen::MatrixXd centers2(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels) {

        // --- Input Validation ---
        if (data.rows() != labels.size()) {
            // In a real application, throwing an exception would be better.
            std::cerr << "Error in centers2: Data rows (" << data.rows()
                      << ") does not match labels size (" << labels.size() << ")." << std::endl;
            // Return an empty matrix to signal failure
            return Eigen::MatrixXd();
        }

        if (data.rows() == 0) {
            std::cerr << "Error in centers2: Input data is empty." << std::endl;
            return Eigen::MatrixXd();
        }

        // --- Initialization ---

        // Find the number of clusters (k).
        // Assumes labels are 0-indexed (0, 1, ..., k-1).
        // .maxCoeff() finds the highest label value. Add 1 for the total count.
        const int num_clusters = labels.maxCoeff() + 1;

        // Get the number of features (dimensions) from the data.
        const long n_cols = data.cols();

        // Initialize a matrix to store the *sum* of vectors for each cluster.
        // We'll divide by the count later to get the mean.
        Eigen::MatrixXd centers = Eigen::MatrixXd::Zero(num_clusters, n_cols);

        // Initialize a vector to store the count of points in each cluster.
        Eigen::VectorXi cluster_counts = Eigen::VectorXi::Zero(num_clusters);

        // --- Calculation (Single Pass) ---

        // Iterate through each data point (row) once.
        for (long i = 0; i < data.rows(); ++i) {
            // Get the cluster ID for the i-th data point.
            const int cluster_id = labels(i);

            // Add the i-th data point's feature vector (data.row(i))
            // to the running sum for its cluster (centers.row(cluster_id)).
            centers.row(cluster_id) += data.row(i);

            // Increment the counter for that cluster.
            cluster_counts(cluster_id)++;
        }

        // --- Calculate Mean ---

        // Now, divide the sums by the counts to get the final mean (centroid)
        for (int i = 0; i < num_clusters; ++i) {
            if (cluster_counts(i) > 0) {
                // Use component-wise division
                centers.row(i) /= cluster_counts(i);
            }
            // If cluster_counts(i) == 0, the center for that cluster
            // remains a zero vector, which is a reasonable default for an empty cluster.
        }

        return centers;
    }
};

