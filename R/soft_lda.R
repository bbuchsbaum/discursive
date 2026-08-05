#' Compute Weighted Group Means
#'
#' Computes one feature mean per row of a class-by-observation weight matrix.
#'
#' @param X Numeric matrix with observations in rows and features in columns.
#' @param F Nonnegative class-by-observation weight matrix. Every row must
#'   have positive mass.
#'
#' @return A class-by-feature matrix of weighted means.
#' @export
weighted_group_means <- function(X, F) {
  X <- as.matrix(X)
  F <- as.matrix(F)
  if (!is.numeric(X) || !is.numeric(F) || ncol(F) != nrow(X)) {
    stop("'X' and 'F' must be numeric matrices with ncol(F) equal to nrow(X).")
  }
  if (any(!is.finite(X)) || any(!is.finite(F)) || any(F < 0)) {
    stop("'X' must be finite and 'F' must contain finite nonnegative weights.")
  }
  weight_sums <- rowSums(F)
  if (any(weight_sums <= 0)) {
    stop("Each row of 'F' must have positive weight mass.")
  }

  means <- F %*% X
  means <- sweep(means, 1L, weight_sums, FUN = "/")
  rownames(means) <- rownames(F)
  colnames(means) <- colnames(X)
  means
}


#' Soft-Label Linear Discriminant Projection
#'
#' Fits the SL-LDA projection of Zhao et al. (2014) from nonnegative soft
#' class memberships. The estimator is distinct from [em_soft_lda()], which
#' fits a Gaussian evidential-EM classifier.
#'
#' @param X Numeric matrix or data frame with observations in rows and features
#'   in columns.
#' @param C Numeric membership matrix with one row per observation and at least
#'   two class columns. Entries must be nonnegative; every row and column must
#'   have positive mass.
#' @param preproc A fitted or unfitted preprocessing specification from the
#'   multivarious package. The default performs no preprocessing.
#' @param dp Optional positive integer number of numerical principal components
#'   to retain. The default uses all available nonzero components.
#' @param di Optional positive integer number of within-class modes used for
#'   whitening. The default uses the positive numerical rank.
#' @param dl Optional positive integer number of discriminant components. The
#'   default is limited by the positive between-class rank and `ncol(C) - 1`.
#' @param alpha Nonnegative ridge added to the within-class scatter.
#' @param tol Relative numerical-rank tolerance in `(0, 1)`.
#'
#' @return A [multivarious::discriminant_projector()] with class `"soft_lda"`.
#'   It includes the complete `memberships`, selected `eigenvalues` and
#'   `within_eigenvalues`, PCA diagnostics, soft scatter matrices, and the
#'   dimensions `dp`, `di`, and `dl`.
#'
#' @details
#' Let \eqn{F=C^T}, \eqn{E=\mathrm{diag}(C\mathbf{1})}, and
#' \eqn{G=\mathrm{diag}(C^T\mathbf{1})}. In the PCA score space, this
#' implementation forms
#' \deqn{S_w = X^T(E-F^T G^{-1}F)X}
#' and
#' \deqn{S_b = X^T(F^T G^{-1}F -
#' E\mathbf{1}\mathbf{1}^T E / \mathbf{1}^T E\mathbf{1})X.}
#'
#' The eigenvalue-selection convention is part of the contract: `di` retains
#' the largest strictly positive eigenvalues of \eqn{S_w+\alpha I}, then
#' whitens those modes. It does not select the smallest eigenvalues or implement
#' null-space LDA. The final directions are the largest positive generalized
#' eigenmodes of \eqn{S_b} relative to the retained within-class subspace.
#'
#' @references
#' Zhao, M., Zhang, Z., Chow, T. W. S., and Li, B. (2014). A general soft
#' label based Linear Discriminant Analysis for semi-supervised dimension
#' reduction. *Neural Networks*, 55, 83--97.
#'
#' @seealso [em_soft_lda()], [pca_lda()]
#' @importFrom multivarious pass fit transform
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' C <- model.matrix(~ iris$Species - 1)
#' fit <- soft_lda(X, C, dl = 2)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
soft_lda <- function(X,
                     C,
                     preproc = pass(),
                     dp = NULL,
                     di = NULL,
                     dl = NULL,
                     alpha = 0,
                     tol = 1e-6) {
  X <- as.matrix(X)
  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (!is.matrix(C) || !is.numeric(C) || nrow(C) != nrow(X) || ncol(C) < 2L) {
    stop("'C' must be a numeric matrix with one row per observation and at least two columns.")
  }
  if (any(!is.finite(C)) || any(C < 0)) {
    stop("'C' must contain only finite nonnegative memberships.")
  }

  row_mass <- rowSums(C)
  class_mass <- colSums(C)
  if (any(row_mass <= 0)) {
    stop("Every row of 'C' must have positive membership mass.")
  }
  if (any(class_mass <= 0)) {
    stop("Every column of 'C' must have positive membership mass.")
  }

  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  alpha <- .check_finite_scalar(alpha, "alpha", lower = 0)
  max_pca_dimension <- min(ncol(X), nrow(X) - 1L)
  if (!is.null(dp)) {
    dp <- .check_whole_number(dp, "dp", lower = 1L, upper = max_pca_dimension)
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  pca_center <- colMeans(Xp)
  centered <- sweep(Xp, 2L, pca_center, FUN = "-")
  pca_svd <- svd(centered, nu = 0L, nv = min(dim(centered)))
  pca_cutoff <- tol * max(pca_svd$d)
  pca_rank <- sum(pca_svd$d > pca_cutoff)
  if (pca_rank == 0L) {
    stop("The preprocessed data have zero numerical PCA rank.")
  }
  if (is.null(dp)) {
    dp <- pca_rank
  } else if (dp > pca_rank) {
    stop(sprintf("'dp' (%d) exceeds the numerical PCA rank (%d).", dp, pca_rank))
  }
  pca_vectors <- pca_svd$v[, seq_len(dp), drop = FALSE]
  pca_scores <- centered %*% pca_vectors

  # These reduced formulas are algebraically identical to the dense E/F/G
  # scatter definitions but avoid materializing n-by-n matrices.
  weighted_total <- crossprod(pca_scores, pca_scores * row_mass)
  class_totals <- crossprod(C, pca_scores)
  class_second <- crossprod(
    sweep(class_totals, 1L, sqrt(class_mass), FUN = "/")
  )
  weighted_sum <- colSums(pca_scores * row_mass)
  total_mass <- sum(row_mass)
  within_scatter <- weighted_total - class_second
  between_scatter <- class_second - tcrossprod(weighted_sum) / total_mass
  within_scatter <- 0.5 * (within_scatter + t(within_scatter))
  between_scatter <- 0.5 * (between_scatter + t(between_scatter))
  regularized_within <- within_scatter + alpha * diag(dp)

  within_decomp <- eigen(regularized_within, symmetric = TRUE)
  within_max <- max(within_decomp$values)
  if (!is.finite(within_max) || within_max <= 0) {
    stop("The soft within-class scatter has no positive numerical rank.")
  }
  within_rank <- sum(within_decomp$values > tol * within_max)
  if (is.null(di)) {
    di <- within_rank
  } else {
    di <- .check_whole_number(di, "di", lower = 1L, upper = dp)
    if (di > within_rank) {
      stop(sprintf("'di' (%d) exceeds the positive within-class rank (%d).", di, within_rank))
    }
  }
  within_values <- within_decomp$values[seq_len(di)]
  whitening <- sweep(
    within_decomp$vectors[, seq_len(di), drop = FALSE],
    2L, sqrt(within_values), FUN = "/"
  )

  whitened_between <- crossprod(whitening, between_scatter %*% whitening)
  whitened_between <- 0.5 * (whitened_between + t(whitened_between))
  between_decomp <- eigen(whitened_between, symmetric = TRUE)
  between_max <- max(between_decomp$values)
  if (!is.finite(between_max) || between_max <= 0) {
    stop("The soft class means have zero positive between-class scatter rank.")
  }
  between_rank <- sum(between_decomp$values > tol * between_max)
  max_dl <- min(di, ncol(C) - 1L, between_rank)
  if (is.null(dl)) {
    dl <- max_dl
  } else {
    dl <- .check_whole_number(
      dl, "dl", lower = 1L, upper = min(di, ncol(C) - 1L)
    )
    if (dl > between_rank) {
      stop(sprintf("'dl' (%d) exceeds the positive between-class rank (%d).", dl, between_rank))
    }
  }

  discriminant_pca <- whitening %*%
    between_decomp$vectors[, seq_len(dl), drop = FALSE]
  vectors <- pca_vectors %*% discriminant_pca
  signs <- vapply(seq_len(dl), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")

  component_names <- paste0("SLDA", seq_len(dl))
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  score_sdev <- apply(scores, 2L, stats::sd)
  names(score_sdev) <- component_names

  class_names <- colnames(C)
  if (is.null(class_names)) class_names <- as.character(seq_len(ncol(C)))
  labels <- factor(
    class_names[max.col(C, ties.method = "first")], levels = class_names
  )

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = labels,
    classes = "soft_lda",
    memberships = C,
    eigenvalues = between_decomp$values[seq_len(dl)],
    within_eigenvalues = within_values,
    pca_singular_values = pca_svd$d[seq_len(dp)],
    within_scatter = within_scatter,
    between_scatter = between_scatter,
    pca_vectors = pca_vectors,
    pca_center = pca_center,
    dp = dp,
    di = di,
    dl = dl,
    pca_rank = pca_rank,
    within_rank = within_rank,
    between_rank = between_rank,
    alpha = alpha,
    tol = tol
  )
}
