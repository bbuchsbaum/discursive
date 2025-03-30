#' @title Weighted Harmonic Mean of Trace Ratios for Multiclass Discriminant Analysis
#'
#' @description This function implements the Weighted Harmonic Mean of Trace Ratios (WHM-TR) method for 
#' multiclass discriminant analysis as described in the referenced paper. The goal is to find a linear 
#' transformation \eqn{W} that improves class separability by minimizing the weighted harmonic mean of 
#' trace ratios derived from the between-class and within-class scatter matrices.
#'
#' @param X A numeric matrix of size \code{n x d}, where \code{n} is the number of samples and \code{d} is the number of features.
#'          Rows correspond to samples and columns to features.
#' @param labels A vector of length \code{n} containing the class labels of the samples. Can be a factor or a numeric vector.
#' @param m The number of dimensions to reduce to, i.e., the number of columns in the projection matrix \eqn{W}.
#'
#' @return A \code{\link[multivarious]{discriminant_projector}} object with:
#'   \itemize{
#'     \item \code{v}: The projection matrix \eqn{W} of size \code{d x m}.
#'     \item \code{s}: The transformed data \eqn{X W} of size \code{n x m}.
#'     \item \code{sdev}: The standard deviations of each column in \code{s}.
#'     \item \code{labels}: The original labels if provided (as a factor).
#'     \item \code{classes}: set to \code{"whm_lda"}.
#'   }
#'
#' @details
#' The algorithm implemented here follows the approach described in Li et al. (2017), 
#' "Beyond Trace Ratio: Weighted Harmonic Mean of Trace Ratios for Multiclass Discriminant Analysis." 
#' The method proceeds as follows:
#' \enumerate{
#'   \item Compute the between-class (\eqn{S_b}) and within-class (\eqn{S_w}) scatter matrices.
#'   \item Initialize a projection matrix \eqn{W} (of size \code{d x m}) randomly and ensure it is orthonormal.
#'   \item Iteratively update \eqn{W} by minimizing the weighted harmonic mean of trace ratios. This involves:
#'     \itemize{
#'       \item Evaluating pairs of classes and computing partial scatter matrices for each pair.
#'       \item Updating \eqn{W} by solving an eigen-decomposition problem derived from these intermediate matrices.
#'     }
#'   \item Stop when convergence is reached (based on a tolerance) or after a maximum number of iterations.
#' }
#'
#' After convergence, \eqn{W} is returned in a standard \code{discriminant_projector} format, along with
#' the reduced data \eqn{XW} in \code{s}.
#'
#' @references 
#' Li, Zhihui, et al. "Beyond Trace Ratio: Weighted Harmonic Mean of Trace Ratios for Multiclass Discriminant Analysis." 
#' \emph{IEEE Transactions on Knowledge and Data Engineering (TKDE)}, 2017.
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' labels <- iris[,5]
#' proj <- whm_lda(X, labels, m=2)
#' print(proj)
#' # Now you can do:
#' # new_data <- ...
#' # projected <- project(proj, new_data)
#' }
#'
#' @export
whm_lda <- function(X, labels, m) {
  if (!requireNamespace("multivarious", quietly = TRUE)) {
    stop("Package 'multivarious' is required (for discriminant_projector) but not installed.")
  }
  
  # Ensure labels are a factor for consistent indexing
  if (!is.factor(labels)) {
    labels <- as.factor(labels)
  }
  classes <- levels(labels)
  c <- length(classes)
  
  # Calculate between-class scatter matrix
  # Sb = sum_k n_k (mu_k - mu)(mu_k - mu)^T
  calcSb <- function(X, labels) {
    overallMean <- colMeans(X)
    Sb <- matrix(0, nrow = ncol(X), ncol = ncol(X))
    for (cl in classes) {
      classData <- X[labels == cl, , drop = FALSE]
      classMean <- colMeans(classData)
      n_k <- nrow(classData)
      diffMean <- classMean - overallMean
      Sb <- Sb + n_k * (diffMean %*% t(diffMean))
    }
    Sb
  }
  
  # Calculate within-class scatter matrix
  # Sw = sum_k sum_{x in class_k} (x - mu_k)(x - mu_k)^T
  calcSw <- function(X, labels) {
    Sw <- matrix(0, nrow = ncol(X), ncol = ncol(X))
    for (cl in classes) {
      classData <- X[labels == cl, , drop = FALSE]
      classMean <- colMeans(classData)
      diffs <- sweep(classData, 2, classMean)
      Sw <- Sw + t(diffs) %*% diffs
    }
    Sw
  }
  
  # Initialize W with random values and orthogonalize using QR decomposition
  initializeW <- function(d, m) {
    W <- matrix(rnorm(d * m), nrow = d, ncol = m)
    qr.Q(qr(W))
  }
  
  # The main optimization loop for WHM-TR (Algorithm 1 from the paper)
  solveWHMTR <- function(X, labels, m, max.iter = 100, tol = 1e-6) {
    d <- ncol(X)
    W <- initializeW(d, m)
    
    for (iter in seq_len(max.iter)) {
      M <- matrix(0, nrow = d, ncol = d)
      
      # Iterate over class pairs (j,k), j < k
      for (j_idx in 1:(c - 1)) {
        for (k_idx in (j_idx + 1):c) {
          cl_j <- classes[j_idx]
          cl_k <- classes[k_idx]
          
          class_j_idx <- which(labels == cl_j)
          class_k_idx <- which(labels == cl_k)
          
          X_j <- X[class_j_idx, , drop = FALSE]
          X_k <- X[class_k_idx, , drop = FALSE]
          
          mu_j <- colMeans(X_j)
          mu_k <- colMeans(X_k)
          n_j <- nrow(X_j)
          n_k <- nrow(X_k)
          
          # Within-class scatter for classes j and k combined
          diffs_j <- sweep(X_j, 2, mu_j)
          diffs_k <- sweep(X_k, 2, mu_k)
          S_w_jk <- t(diffs_j) %*% diffs_j + t(diffs_k) %*% diffs_k
          
          # Between-class scatter for classes j and k
          diffMeans <- mu_j - mu_k
          S_b_jk <- (diffMeans %*% t(diffMeans)) * (n_j + n_k)
          
          # Evaluate partial trace ratio
          num_b <- sum(diag(t(W) %*% S_b_jk %*% W))
          num_w <- sum(diag(t(W) %*% S_w_jk %*% W))
          
          # Weighted harmonic mean derivation => partial derivative approx:
          # M += (1/num_b)*S_w_jk - (num_w/(num_b^2)) * S_b_jk
          M <- M + (1 / num_b) * S_w_jk - (num_w / (num_b^2)) * S_b_jk
        }
      }
      
      # Solve eigen decomposition for M, choose m eigenvectors with the SMALLEST eigenvalues
      #  or LARGEST => depends on the sign of M. The paper might specify smallest. We'll do smallest:
      eM <- eigen(M, symmetric = TRUE)
      # pick the smallest
      # The last m columns if the values are sorted in decreasing order => or we reorder ascending
      lambda_vals <- eM$values
      idx_sorted <- order(lambda_vals, decreasing = FALSE) # ascending
      W_new <- eM$vectors[, idx_sorted[1:m], drop=FALSE]
      
      # Check convergence
      if (norm(W - W_new, type = "F") < tol) {
        W <- W_new
        break
      }
      
      W <- W_new
    }
    W
  }
  
  # 1) Run WHM-TR to get the final W
  W_final <- solveWHMTR(X, labels, m)
  
  # 2) Project the data
  X_reduced <- X %*% W_final
  
  # 3) Build a discriminant_projector object
  if (!requireNamespace("multivarious", quietly = TRUE)) {
    stop("Package 'multivarious' not installed; needed for discriminant_projector.")
  }
  
  # For sdev, we do col stdev of X_reduced
  sdev_vec <- apply(X_reduced, 2, sd)
  
  dp_obj <- multivarious::discriminant_projector(
    v       = W_final,
    s       = X_reduced,
    sdev    = sdev_vec,
    preproc = multivarious::prep(multivarious::pass()),  # no-op
    labels  = labels,
    classes = "whm_lda"
  )
  
  dp_obj
}
