.is_finite_numeric_scalar <- function(x) {
  is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x)
}

.is_finite_positive_integer <- function(x) {
  .is_finite_numeric_scalar(x) && x > 0 && x == round(x)
}

#' @title Apply PCA for Preprocessing (e.g., Null Space Removal)
#' @description Preprocesses the data matrix using Principal Component Analysis (PCA).
#'   Retains a specified number of principal components, ensuring that at least
#'   `var_retained` proportion of the total variance is captured. This step can
#'   help reduce dimensionality and remove potential null space issues.
#' @param X A numeric matrix or data frame of size n x d (n samples, d features).
#' @param var_retained The minimum proportion of variance to retain (numeric, default is 0.95).
#'   Must be finite and between 0 and 1.
#' @return A list containing:
#'   \item{X_pca}{The transformed data matrix after PCA (n x d_pca), where d_pca is
#'           the number of components retained.}
#'   \item{rot}{The PCA rotation (loadings) matrix (d x d_pca) that transforms
#'           the original features to the principal components.}
#'   \item{fit}{The original PCA fit object obtained from `multivarious::pca`.}
#' @importFrom multivarious pca pass
#' @keywords internal
apply_pca <- function(X, var_retained = 0.95) {
  assertthat::assert_that(is.matrix(X) || is.data.frame(X), msg = "X must be a matrix or data frame.")
  X <- as.matrix(X)
  assertthat::assert_that(is.numeric(X), msg = "X must be numeric.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(
    .is_finite_numeric_scalar(var_retained) && var_retained > 0 && var_retained <= 1,
    msg = "var_retained must be a finite numeric scalar between 0 and 1."
  )
  assertthat::assert_that(
    length(X) > 0L && max(abs(X)) > 0,
    msg = "X must contain non-zero variation after preprocessing."
  )

  pca_result <- multivarious::pca(X, preproc = multivarious::pass())

  singular_scale <- max(abs(pca_result$sdev))
  assertthat::assert_that(
    is.finite(singular_scale) && singular_scale > 0,
    msg = "X must contain non-zero variation after preprocessing."
  )

  # Scaling before squaring avoids underflow and does not affect variance ratios.
  singular_values_sq <- (pca_result$sdev / singular_scale)^2
  var_explained <- singular_values_sq / sum(singular_values_sq)
  cum_var <- cumsum(var_explained)
  ncomp <- which(cum_var >= var_retained)[1]
  if (is.na(ncomp)) {
    ncomp <- length(var_explained)
  }

  ncomp <- min(ncomp, length(pca_result$sdev))
  X_pca <- pca_result$s[, 1:ncomp, drop = FALSE]
  rotation_matrix <- pca_result$v[, 1:ncomp, drop = FALSE]

  list(X_pca = X_pca, rot = rotation_matrix, fit = pca_result)
}


.validate_allda_neighborhood <- function(y, k, n = length(y)) {
  assertthat::assert_that(
    .is_finite_positive_integer(k),
    msg = "k must be a finite positive integer."
  )
  assertthat::assert_that(!is.null(y) && length(y) == n, msg = "Length of y must match number of rows in X.")
  assertthat::assert_that(!anyNA(y), msg = "y must not contain missing values.")

  y <- droplevels(as.factor(y))
  class_counts <- table(y)
  assertthat::assert_that(length(class_counts) >= 2, msg = "y must contain at least two classes.")

  min_class_size <- min(class_counts)
  assertthat::assert_that(
    k < min_class_size,
    msg = paste0("k must be less than the size of every class; minimum class size is ", min_class_size, ".")
  )

  invisible(class_counts)
}


.within_class_knn <- function(X, y, k, square = TRUE) {
  y <- droplevels(as.factor(y))
  n <- nrow(X)
  indices <- matrix(NA_integer_, nrow = n, ncol = k)
  distances <- matrix(NA_real_, nrow = n, ncol = k)

  for (level in levels(y)) {
    rows <- which(y == level)
    X_class <- X[rows, , drop = FALSE]
    nn_result <- Rnanoflann::nn(
      data = X_class,
      points = X_class,
      k = k + 1,
      method = "euclidean",
      search = "standard",
      eps = 0.0,
      square = square,
      trans = TRUE
    )

    for (local_i in seq_along(rows)) {
      candidate_local <- nn_result$indices[local_i, ]
      candidate_distances <- nn_result$distances[local_i, ]
      keep <- candidate_local != local_i
      candidate_local <- candidate_local[keep]
      candidate_distances <- candidate_distances[keep]

      if (length(candidate_local) < k) {
        stop("Unable to find k non-self neighbors within class '", level, "'.")
      }

      take <- seq_len(k)
      global_i <- rows[local_i]
      indices[global_i, ] <- rows[candidate_local[take]]
      distances[global_i, ] <- candidate_distances[take]
    }
  }

  list(indices = indices, distances = distances)
}


#' @title Initialize Similarity Matrix for ALLDA
#' @description Initializes the similarity matrix S based on k-nearest neighbors (KNN).
#'   Each point `i` is connected to its `k` nearest neighbors `j` with a
#'   weight `S_ij = 1/k`. Neighbor search is restricted to points with the same
#'   class label as `i`.
#' @param X A numeric data matrix of size n x d (n samples, d features), typically
#'   the PCA-reduced data.
#' @param y Label vector (length n). The initial graph connects only same-class
#'   neighbors.
#' @param ncomp Reduced dimension (integer). Not used in this specific initialization
#'   function, but included for API consistency.
#' @param k The number of neighbors (integer) to consider for each point.
#' @return A list containing the initialized similarity matrix `S` (n x n) as a
#'   sparse `Matrix` (k non-zeros per row).
#' @keywords internal
initialize_parameters <- function(X, y, ncomp, k) {
  n <- nrow(X)  # Number of samples

  # Input checks
  assertthat::assert_that(
    .is_finite_positive_integer(k) && k < n,
    msg = "k must be a finite positive integer less than the number of samples."
  )
  .validate_allda_neighborhood(y, k, n)

  # S has only n*k non-zeros, so build it sparse: O(n*k) storage instead of O(n^2)
  knn <- .within_class_knn(X, y, k, square = TRUE)
  S <- Matrix::sparseMatrix(
    i = rep(seq_len(n), times = k),
    j = as.vector(knn$indices),
    x = 1 / k,
    dims = c(n, n)
  )
  list(S = S)
}


#' @title Update Laplacian Matrix
#' @description Computes the graph Laplacian matrix L_A based on the similarity matrix S.
#'   Uses the formula `L_A = D_A - (S + t(S)) / 2`, where D_A is the diagonal
#'   degree matrix `diag(rowSums((S + t(S)) / 2))`. This symmetric form is common
#'   in spectral methods.
#' @param S A numeric similarity matrix (n x n). Note that S might not be symmetric.
#' @return The computed Laplacian matrix L_A (n x n), which will be symmetric.
#' @importFrom Matrix diag
#' @keywords internal
update_laplacian <- function(S) {
  # Symmetrize the similarity matrix first
  S_sym <- (S + t(S)) / 2

  L_A <- -S_sym
  diag(L_A) <- diag(L_A) + rowSums(S_sym)
  L_A
}


.prepare_allda_whitener <- function(X, reg = 1e-5) {
  assertthat::assert_that(is.matrix(X) && is.numeric(X), msg = "X must be a numeric matrix.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(
    .is_finite_numeric_scalar(reg) && reg >= 0,
    msg = "reg must be a finite non-negative numeric scalar."
  )

  d <- ncol(X)
  data_scale <- max(abs(X))
  assertthat::assert_that(
    is.finite(data_scale) && data_scale > 0,
    msg = "X must contain non-zero variation before whitening."
  )

  # Work at a normalized scale to avoid overflow/underflow. Dividing the final
  # eigenvectors by data_scale restores normalization in the original units.
  X_scaled <- X / data_scale
  St_scaled <- crossprod(X_scaled)
  scatter_scale <- mean(diag(St_scaled))
  assertthat::assert_that(
    is.finite(scatter_scale) && scatter_scale > 0,
    msg = "The total scatter of X must be positive."
  )
  ridge_scaled <- reg * scatter_scale
  St_reg_scaled <- as.matrix(St_scaled + ridge_scaled * diag(d))

  chol_St <- tryCatch(
    chol(St_reg_scaled),
    error = function(e) {
      stop(
        "The relatively regularized total-scatter matrix is not positive definite. ",
        "Consider increasing 'reg'. Error: ", e$message,
        call. = FALSE
      )
    }
  )

  list(
    X_scaled = X_scaled,
    data_scale = data_scale,
    St_scaled = St_scaled,
    St_reg_scaled = St_reg_scaled,
    St_inv_half = backsolve(chol_St, diag(d)),
    scatter_scale = scatter_scale,
    ridge_scaled = ridge_scaled,
    reg = reg
  )
}


#' @title Update Transformation Matrix W
#' @description Solves the generalized eigenvalue problem
#'   `t(X) %*% L_A %*% X %*% W = lambda %*% (t(X) %*% X + rho I) %*% W`,
#'   where `rho = reg * mean(diag(t(X) %*% X))`,
#'   to find the optimal transformation matrix W that minimizes the ALLDA objective.
#'   It finds the eigenvectors corresponding to the smallest eigenvalues.
#' @param X Data matrix of size n x d (n samples, d features), typically PCA-reduced.
#' @param L_A Laplacian matrix of size n x n.
#' @param ncomp The desired reduced dimension (number of eigenvectors to select).
#' @param reg Non-negative relative regularization multiplier. The diagonal ridge
#'   is `reg * mean(diag(t(X) %*% X))`, making it independent of measurement
#'   units (default is 1e-5).
#' @param whitener Optional cached whitening information created by
#'   `.prepare_allda_whitener()` for the same `X` and `reg`.
#' @return The transformation matrix W (d x ncomp) containing the selected eigenvectors
#'   as columns, normalized under the relatively regularized total scatter.
#' @importFrom Matrix diag
#' @importFrom RSpectra eigs_sym
#' @keywords internal
update_W <- function(X, L_A, ncomp, reg = 1e-5, whitener = NULL) {
  assertthat::assert_that(is.matrix(X) && is.numeric(X), msg = "X must be a numeric matrix.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(
    .is_finite_positive_integer(ncomp) && ncomp <= ncol(X),
    msg = "ncomp must be a finite positive integer no larger than ncol(X)."
  )
  assertthat::assert_that(
    length(dim(L_A)) == 2L && all(dim(L_A) == nrow(X)),
    msg = "L_A must be a square matrix with one row per observation in X."
  )
  assertthat::assert_that(all(is.finite(L_A)), msg = "L_A must contain only finite values.")

  if (is.null(whitener)) {
    whitener <- .prepare_allda_whitener(X, reg)
  } else {
    assertthat::assert_that(
      is.list(whitener) &&
        identical(dim(whitener$X_scaled), dim(X)) &&
        .is_finite_numeric_scalar(whitener$data_scale),
      msg = "whitener must be created for an X with the same dimensions."
    )
    assertthat::assert_that(
      isTRUE(all.equal(reg, whitener$reg)),
      msg = "whitener must use the same reg value supplied to update_W()."
    )
  }

  d <- ncol(X)
  X_scaled <- whitener$X_scaled

  # Calculate the graph-regularized scatter matrix Xt_LA_X = t(X) %*% L_A %*% X
  # (as.matrix: L_A may be sparse, but the d x d result is small and dense)
  Xt_LA_X <- as.matrix(crossprod(X_scaled, L_A %*% X_scaled))
  Xt_LA_X <- 0.5 * (Xt_LA_X + t(Xt_LA_X))

  St_inv_half <- whitener$St_inv_half
  whitened <- crossprod(St_inv_half, Xt_LA_X %*% St_inv_half)
  whitened <- 0.5 * (whitened + t(whitened))

  eig <- tryCatch({
    if (d <= 2 || ncomp >= (d - 1)) {
      eigen(whitened, symmetric = TRUE)
    } else {
      RSpectra::eigs_sym(whitened, k = ncomp, sigma = 0)
    }
  }, error = function(e) {
    eigen(whitened, symmetric = TRUE)
  })

  # eigs_sym can return normally with fewer than ncomp converged eigenpairs;
  # fall back to the dense solver in that case
  if (length(eig$values) < ncomp) {
    eig <- eigen(whitened, symmetric = TRUE)
  }

  ord <- order(eig$values)
  
  # Restore B-normalization in the original units after the scaled solve.
  (St_inv_half %*% eig$vectors[, ord[seq_len(ncomp)], drop = FALSE]) /
    whitener$data_scale
}


#' @title Update Similarity Matrix S
#' @description Updates the similarity matrix S based on the current projection W.
#'   For each point `i`, it finds the `k` nearest neighbors `j` in the transformed
#'   space `X %*% W`, restricted to same-class neighbors. The similarity `S_ij`
#'   is calculated based on the squared distances
#'   `d_ij` in this space using the formula:
#'   `S_ij = d_ij^(1/(1-r)) / sum_{l in KNN(i)} d_il^(1/(1-r))` if `j` is a neighbor of `i`,
#'   and `S_ij = 0` otherwise.
#' @param X Data matrix of size n x d (n samples, d features), typically PCA-reduced.
#' @param W Transformation matrix (d x ncomp).
#' @param y Label vector of length n. Neighbor search is restricted to points in
#'   the same class.
#' @param k Number of neighbors (integer).
#' @param r Parameter controlling the influence of distances (numeric, must be > 1).
#'   Larger `r` gives more equal weights to neighbors.
#' @param epsilon Relative threshold used to identify effectively duplicate
#'   neighbors within each row (default is 1e-10). It is not added to distances,
#'   so the result is invariant to global changes of measurement units.
#' @return Updated similarity matrix S (n x n) as a sparse `Matrix`
#'   (at most k non-zeros per row; positive-distance neighbors receive zero
#'   limiting weight when duplicate neighbors are present).
#' @importFrom Rnanoflann nn
#' @keywords internal
update_similarity <- function(X, W, y, k, r, epsilon = 1e-10) {
  n <- nrow(X) # Number of samples

  # Input checks
  assertthat::assert_that(
    .is_finite_numeric_scalar(r) && r > 1,
    msg = "r must be a finite numeric scalar greater than 1."
  )
  assertthat::assert_that(
    .is_finite_positive_integer(k) && k < n,
    msg = "k must be a finite positive integer less than the number of samples."
  )
  assertthat::assert_that(
    .is_finite_numeric_scalar(epsilon) && epsilon >= 0 && epsilon < 1,
    msg = "epsilon must be a finite numeric scalar in [0, 1)."
  )
  .validate_allda_neighborhood(y, k, n)

  # Pre-calculate the exponent for efficiency
  one_over_1_minus_r <- 1 / (1 - r) # This will be negative since r > 1

  # Project data into the current subspace defined by W
  X_transformed <- X %*% W

  knn <- .within_class_knn(X_transformed, y, k, square = TRUE)

  distances <- pmax(knn$distances, 0)

  # d^(1/(1-r)) has an infinite limiting value at d = 0. If a row has
  # duplicate neighbors, distribute all mass uniformly among those duplicates.
  # The threshold is relative to the row scale, preserving unit invariance.
  row_scale <- apply(distances, 1, max)
  duplicate_threshold <- epsilon * row_scale
  duplicates <- distances <= duplicate_threshold
  has_duplicates <- rowSums(duplicates) > 0
  weights_normalized <- matrix(0, nrow = n, ncol = k)

  if (any(has_duplicates)) {
    duplicate_weights <- duplicates[has_duplicates, , drop = FALSE]
    weights_normalized[has_duplicates, ] <-
      duplicate_weights / rowSums(duplicate_weights)
  }

  positive_rows <- !has_duplicates
  if (any(positive_rows)) {
    positive_distances <- distances[positive_rows, , drop = FALSE]
    log_weights <- one_over_1_minus_r * log(positive_distances)
    log_weights <- log_weights - apply(log_weights, 1, max)
    weights <- exp(log_weights)
    weights_normalized[positive_rows, ] <- weights / rowSums(weights)
  }

  # S has only n*k non-zeros, so build it sparse from (row, neighbor, weight) triplets
  Matrix::drop0(Matrix::sparseMatrix(
    i = rep(seq_len(n), times = k),
    j = as.vector(knn$indices),
    x = as.vector(weights_normalized),
    dims = c(n, n)
  ))
}


.validate_allda_centering <- function(X) {
  data_scale <- max(abs(X))
  if (!is.finite(data_scale) || data_scale == 0) {
    return(invisible(TRUE))
  }

  centering_error <- max(abs(colMeans(X)))
  relative_tolerance <- sqrt(.Machine$double.eps) * data_scale
  assertthat::assert_that(
    centering_error <= relative_tolerance,
    msg = paste0(
      "ALLDA requires pre-centered data after preprocessing. ",
      "Use multivarious::center(), or provide pre-centered X when using multivarious::pass()."
    )
  )
  invisible(TRUE)
}


.allda_objective <- function(L_A, W, whitener) {
  W_scaled <- W * whitener$data_scale
  projected <- whitener$X_scaled %*% W_scaled
  as.numeric(sum(projected * (L_A %*% projected)))
}


#' @title Adaptive Local Linear Discriminant Analysis (ALLDA)
#' @description Performs dimensionality reduction using the ALLDA algorithm, which
#'   adaptively learns a similarity structure while performing discriminant analysis.
#'   It iteratively updates a similarity matrix S and a projection matrix W.
#'
#' @param X A numeric data matrix of size n x d (n samples, d features).
#' @param y A vector of length n containing the class labels for each sample.
#'   Labels are used to restrict adaptive neighborhoods to within-class points.
#' @param ncomp The target dimensionality (integer). Must be less than the number
#'   of features retained after the initial PCA preprocessing step.
#' @param k The number of same-class neighbors to consider when constructing and
#'   updating the similarity matrix S (integer). Must be smaller than every class
#'   size.
#' @param r Parameter controlling the distance weighting in the similarity update
#'   (numeric, default is 2). Must be greater than 1. Larger values lead to more
#'   uniform weights among neighbors.
#' @param preproc A preprocessing object from `multivarious` specifying steps like
#'   centering or scaling to be applied before PCA and ALLDA. Defaults to centering
#'   using `multivarious::center()`. ALLDA requires centered total scatter; use
#'   `multivarious::pass()` only when `X` is already column-centered.
#' @param max_iter Maximum number of iterations for the S and W updates (integer,
#'   default is 30).
#' @param tol Convergence tolerance (numeric, default is 1e-4). Iteration stops
#'   if the Frobenius norm of the change in S is less than `tol`.
#' @param var_retained Proportion of variance to retain during the initial PCA
#'   step (numeric, default is 0.95). Helps stabilize the process and remove
#'   null space.
#' @param reg Non-negative relative regularization multiplier used during the W
#'   update (numeric, default is 1e-5). The diagonal ridge is
#'   `reg * mean(diag(t(X) %*% X))`, so changing measurement units does not
#'   change the fitted subspace.
#'
#' @return An S3 object of class `discriminant_projector` (from `multivarious`),
#'   inheriting from `projector`. This object contains:
#'   \item{W}{The final projection matrix (d x ncomp) mapping from the preprocessed
#'            original feature space to the reduced dimension space.}
#'   \item{scores}{The data projected onto the final `W` (n x ncomp).}
#'   \item{sdev}{Standard deviations of the scores (related to eigenvalues).}
#'   \item{preproc}{The preprocessing object used.}
#'   \item{pca_result}{The result from the initial `apply_pca` step (list).}
#'   \item{S}{The final similarity matrix (n x n, sparse `Matrix`).}
#'   \item{W_pca}{The final projection in PCA coordinates. It solves the
#'            generalized eigenproblem defined by the returned `S`.}
#'   \item{converged}{Whether the last similarity update was smaller than `tol`.}
#'   \item{final_delta}{Frobenius norm of the final similarity update.}
#'   \item{objective}{Final graph objective for the returned coherent `(W, S)` pair.}
#'   \item{ncomp}{The target dimensionality.}
#'   \item{classes}{The specific class name "allda".}
#'   \item{...}{Other slots inherent to `discriminant_projector`.}
#'   This object can be used with `predict()` from `multivarious` to project new data.
#'
#' @references Nie, F., Wang, Z., Wang, R., Wang, Z., & Li, X. (2020).
#'   Adaptive local linear discriminant analysis.
#'   ACM Transactions on Knowledge Discovery from Data (TKDD), 14(1), 1-19.
#'   \doi{10.1145/3369870}.
#'
#' @examples
#' # Example using iris data
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris$Species
#'
#' # Reduce to 2 dimensions, using 7 neighbors
#' # Use tryCatch as ALLDA can sometimes be sensitive to parameters
#' result <- tryCatch({
#'   allda(X, y, ncomp = 2, k = 7, r = 2, var_retained = 0.99)
#' }, error = function(e) {
#'   warning("ALLDA failed with error: ", e$message)
#'   NULL
#' })
#'
#' if (!is.null(result)) {
#'   print(summary(result))
#'   # W_final <- result$W # or coefficients(result)
#'   # S_final <- result$S # Access custom slots if needed
#'   # scores <- scores(result)
#'   # plot(scores, col = as.factor(y), main = "ALLDA Projection")
#' }
#'
#' @importFrom multivarious discriminant_projector center pass
#' @importFrom assertthat assert_that
#' @import Matrix
#'
#' @export
allda <- function(X, y, ncomp, k, r = 2, preproc = multivarious::center(),
                  max_iter = 30, tol = 1e-4, var_retained = 0.95, reg = 1e-5) {
  assertthat::assert_that(is.matrix(X) || is.data.frame(X), msg = "X must be a matrix or data frame.")
  X <- as.matrix(X)
  assertthat::assert_that(is.numeric(X), msg = "X must be numeric.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(nrow(X) > 1L && ncol(X) > 1L, msg = "X must have at least two rows and two columns.")
  n_samples <- nrow(X)
  assertthat::assert_that(
    .is_finite_positive_integer(ncomp),
    msg = "ncomp must be a finite positive integer."
  )
  assertthat::assert_that(
    .is_finite_positive_integer(k) && k < n_samples,
    msg = "k must be a finite positive integer less than the number of samples."
  )
  assertthat::assert_that(
    .is_finite_numeric_scalar(r) && r > 1,
    msg = "r must be a finite numeric scalar greater than 1."
  )
  assertthat::assert_that(
    .is_finite_positive_integer(max_iter),
    msg = "max_iter must be a finite positive integer."
  )
  assertthat::assert_that(
    .is_finite_numeric_scalar(tol) && tol > 0,
    msg = "tol must be a finite positive numeric scalar."
  )
  assertthat::assert_that(
    .is_finite_numeric_scalar(var_retained) && var_retained > 0 && var_retained <= 1,
    msg = "var_retained must be a finite numeric scalar between 0 and 1."
  )
  assertthat::assert_that(
    .is_finite_numeric_scalar(reg) && reg >= 0,
    msg = "reg must be a finite non-negative numeric scalar."
  )
  .validate_allda_neighborhood(y, k, n_samples)

  # 1. Preprocessing (e.g., centering, scaling) using multivarious framework
  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  assertthat::assert_that(
    is.numeric(Xp) && identical(dim(Xp), dim(X)),
    msg = "preproc must return a numeric matrix with the same dimensions as X."
  )
  assertthat::assert_that(all(is.finite(Xp)), msg = "preproc must return only finite values.")
  .validate_allda_centering(Xp)
  
  # 2. Apply PCA for dimensionality reduction / null space removal
  pca_result <- apply_pca(Xp, var_retained)
  X_pca <- pca_result$X_pca # Data in PCA space (n x d_pca)
  d_pca <- ncol(X_pca)
  
  # Check if requested ncomp is valid after PCA
  assertthat::assert_that(ncomp < d_pca,
                          msg = paste0("ncomp (", ncomp, ") must be less than the number of retained PCA components (", d_pca, "). Increase var_retained or decrease ncomp."))
  
  # 3. Initialize parameters (Similarity matrix S) using PCA-reduced data
  # The initial graph connects each point to its k nearest same-class neighbors
  init_params <- initialize_parameters(X_pca, y, ncomp, k)
  S <- init_params$S # Initial similarity matrix (n x n, sparse)

  # X_pca and reg are fixed across iterations, so cache the total-scatter
  # whitening rather than repeating its crossproduct and Cholesky factorization.
  whitener <- .prepare_allda_whitener(X_pca, reg)

  # 4. Iterative updates for L_A, W, and S
  iter <- 0
  converged <- FALSE
  delta_S_norm <- Inf
  for (iter in seq_len(max_iter)) {
    # Update Laplacian L_A based on current S
    L_A <- update_laplacian(S)

    # Update projection W based on current X_pca and L_A
    # W here is d_pca x ncomp
    W <- update_W(X_pca, L_A, ncomp, reg, whitener = whitener)

    # Update similarity matrix S based on current X_pca and W
    S_new <- update_similarity(X_pca, W, y, k, r)

    # Check for convergence using the Frobenius norm of the change in S
    # (computed directly so it works for both sparse and dense S)
    delta_S_norm <- sqrt(sum((S_new - S)^2))

    if (delta_S_norm < tol) {
      converged <- TRUE
      S <- S_new
      break
    }

    # Prepare for next iteration
    S <- S_new
  }

  if (!converged) {
    warning("ALLDA did not converge within ", max_iter, " iterations. Tolerance = ", tol, ", Final change = ", delta_S_norm)
  }

  # The loop's W was fitted to the graph preceding its final S update. Refit W
  # once against the returned S so the result is internally coherent even when
  # max_iter is exhausted before convergence.
  final_L_A <- update_laplacian(S)
  W <- update_W(X_pca, final_L_A, ncomp, reg, whitener = whitener)
  objective <- .allda_objective(final_L_A, W, whitener)

  # 5. Map the projection W back to the original (preprocessed) space
  # pca_result$rot is d x d_pca; W is d_pca x ncomp
  # W_final is d x ncomp (projection from original preprocessed space)
  W_final <- pca_result$rot %*% W

  # 6. Calculate final scores on the *preprocessed* data Xp
  # This ensures consistency with the multivarious::projector framework
  scores <- Xp %*% W_final

  # 7. Calculate standard deviations of scores (often stored by projectors)
  sds <- apply(scores, 2, sd)

  # 8. Create the final multivarious::discriminant_projector object
  # Store relevant information within the object
  proj <- multivarious::discriminant_projector(
    v = W_final,             # Projection matrix
    s = scores,              # Projected data
    sdev = sds,              # Standard deviation of scores
    preproc = procres,       # Preprocessing steps applied to X
    labels = as.character(y), # Original labels
    ncomp = ncomp,           # Target dimensionality
    pca_result = pca_result, # Store PCA info (optional, but useful)
    S = S,                   # Final similarity matrix (custom slot)
    W_pca = W,               # Projection coherent with S in PCA coordinates
    k = k,                   # Store k (custom slot)
    r = r,                   # Store r (custom slot)
    reg = reg,               # Relative ridge multiplier
    converged = converged,   # Store convergence status (custom slot)
    iterations = iter,       # Store number of iterations (custom slot)
    final_delta = delta_S_norm,
    objective = objective,
    W = W_final,             # Backwards-compatible field name
    scores = scores,         # Backwards-compatible field name
    classes = "allda"        # Specific class identifier
  )

  proj
}
