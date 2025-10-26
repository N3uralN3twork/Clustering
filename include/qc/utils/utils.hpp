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
     * @brief A struct to hold the results of the gss() (Get Sum of Squares) calculation.
     */
    struct GssResult {
        double wgss = 0.0; // Within-cluster Sum of Squares
        double bgss = 0.0; // Between-cluster Sum of Squares
        double tss = 0.0;  // Total Sum of Squares (allmeandist in Python)
        Eigen::MatrixXd centers; // The calculated centroids
    };

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
    static Eigen::MatrixXd get_centroids(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels) {

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

    /**
     * @brief Calculates the total, within-cluster, and between-cluster sum of squares.
     *
     * @param data The input data matrix (n_samples, n_features).
     * @param labels A vector of cluster labels (n_samples, 1).
     * @return GssResult A struct containing wgss, bgss, tss, and the cluster centers.
     */
    static GssResult gss(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels) {

        // --- Step 1: Calculate Centroids ---
        // We re-use the efficient function we already wrote.
        Eigen::MatrixXd centers = get_centroids(data, labels);

        // --- Step 2: Calculate Total Sum of Squares (TSS) ---
        // This is the sum of squared distances from all points to the grand mean.

        // 1. Find the grand mean (mean of all data points)
        // .colwise().mean() computes the mean of each column -> (1, n_features) vector
        Eigen::RowVectorXd allmean = data.colwise().mean();

        // 2. Subtract the grand mean from every row and get the sum of all squared elements.
        // .rowwise() - allmean = broadcasts the subtraction
        // .squaredNorm() = efficient sum of all squared elements in the resulting matrix
        const double tss = (data.rowwise() - allmean).squaredNorm();


        // --- Step 3: Calculate Within-cluster Sum of Squares (WGSS) ---
        // This is the sum of squared distances from each point to its *own* cluster centroid.
        // We can do this in a single pass over the data, just like in centers2.
        double wgss = 0.0;
        for (long i = 0; i < data.rows(); ++i) {
            const int cluster_id = labels(i);

            // (data.row(i) - centers.row(cluster_id)) gives the vector from point to centroid
            // .squaredNorm() gives the squared distance
            wgss += (data.row(i) - centers.row(cluster_id)).squaredNorm();
        }

        // --- Step 4: Calculate Between-cluster Sum of Squares (BGSS) ---
        // We know that TSS = WGSS + BGSS
        const double bgss = tss - wgss;

        return {wgss, bgss, tss, centers};
    }
};

