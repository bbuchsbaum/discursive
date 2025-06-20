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
  if (!is.numeric(m) || length(m) != 1 || m < 1 || m > ncol(X)) {
    stop("'m' must be an integer between 1 and the number of features")
  }

  classes <- levels(labels)
  c <- length(classes)

  # Initialize W with random values and orthogonalize using QR decomposition
  initializeW <- function(d, m) {
    W <- matrix(rnorm(d * m), nrow = d, ncol = m)
    qr.Q(qr(W))
  }

  # The main optimization loop for WHM-TR (Algorithm 1 from the paper)
  solveWHMTR <- function(X, labels, m, max.iter = 100, tol = 1e-6) {
    d <- ncol(X)
    W <- initializeW(d, m)

    class_stats <- lapply(classes, function(cl) {
      idx <- which(labels == cl)
      Xc <- X[idx, , drop = FALSE]
      mu <- colMeans(Xc)
      list(
        n  = nrow(Xc),
        mu = mu,
        Sw = crossprod(sweep(Xc, 2, mu))
      )
    })

    pair_stats <- list()
    for (j_idx in 1:(c - 1)) {
      for (k_idx in (j_idx + 1):c) {
        sj <- class_stats[[j_idx]]
        sk <- class_stats[[k_idx]]
        diff <- sj$mu - sk$mu
        pair_stats[[length(pair_stats) + 1]] <- list(
          Sw = sj$Sw + sk$Sw,
          Sb = tcrossprod(diff) * (sj$n + sk$n)
        )
      }
    }
    
    for (iter in seq_len(max.iter)) {
      M <- matrix(0, nrow = d, ncol = d)
      
      for (pair in pair_stats) {
        num_b <- sum(diag(crossprod(W, pair$Sb %*% W)))
        if (abs(num_b) < .Machine$double.eps) next
        num_w <- sum(diag(crossprod(W, pair$Sw %*% W)))
        M <- M + (1 / num_b) * pair$Sw - (num_w / (num_b^2)) * pair$Sb
      }
      
      # Solve eigen decomposition for M. Li et al. (2017) select the eigenvectors
      # associated with the smallest eigenvalues.
      eM <- eigen(M, symmetric = TRUE)
      # Pick eigenvectors corresponding to the smallest eigenvalues
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
