#' Compute Group Means
#'
#' @param Y A factor with one group label per row of `X`.
#' @param X Numeric matrix with observations in rows.
#'
#' @return A group-by-feature matrix of arithmetic means.
#' @export
group_means <- function(Y, X) {
  X <- as.matrix(X)
  if (!is.factor(Y)) {
    stop("'Y' must be a factor.")
  }
  if (!is.numeric(X) || length(Y) != nrow(X)) {
    stop("'X' must be numeric with one row per element of 'Y'.")
  }
  Y <- droplevels(Y)
  counts <- tabulate(Y, nbins = nlevels(Y))
  means <- rowsum(X, Y, reorder = TRUE)
  means <- sweep(means, 1L, counts, FUN = "/")
  rownames(means) <- levels(Y)
  means
}


#' Fast Orthogonal Linear Discriminant Analysis
#'
#' Computes the OLDA basis directly from compact singular-value decompositions
#' of the total-scatter factor and the whitened class-centroid matrix. Unlike
#' [olda()], this fused implementation does not construct an intermediate ULDA
#' projector and does not delegate to either `ulda()` or `olda()`.
#'
#' @param X Numeric matrix or data frame with observations in rows and features
#'   in columns.
#' @param Y Class labels with length equal to `nrow(X)`.
#' @param preproc A fitted or unfitted preprocessing specification from the
#'   multivarious package. The default centers the predictors.
#' @param reg Nonnegative relative ridge applied to the total-scatter spectrum.
#'   `reg = 0` gives the unregularized Ye (2005) OLDA subspace. Positive values
#'   replace \eqn{\sigma_j^2} by
#'   \eqn{\sigma_j^2 + \mathrm{reg}\,\sigma_1^2} before whitening.
#' @param tol Relative numerical-rank tolerance in `(0, 1)`.
#' @param ncomp Optional positive integer number of components. The default
#'   uses the numerical between-class rank.
#'
#' @return A [multivarious::discriminant_projector()] with class
#'   `"fast_olda"`. Additional fields report the eigenvalues, numerical ranks,
#'   effective ridge, and the direct algorithm used.
#'
#' @details
#' The implementation fuses steps 1--5 of Ye's ULDA algorithm with the final
#' QR decomposition in the OLDA algorithm. For `reg = 0`, its returned basis
#' spans the same subspace as [olda()] up to an arbitrary orthogonal rotation.
#'
#' @references
#' Ye, J. (2005). Characterization of a Family of Algorithms for Generalized
#' Discriminant Analysis on Undersampled Problems. *Journal of Machine Learning
#' Research*, 6, 483--502.
#'
#' @seealso [olda()], [ulda()]
#' @importFrom multivarious center fit transform
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' fit <- fastolda(X, iris$Species, reg = 0, ncomp = 2)
#' crossprod(coef(fit))
fastolda <- function(X,
                     Y,
                     preproc = center(),
                     reg = 0.01,
                     tol = 1e-6,
                     ncomp = NULL) {
  X <- as.matrix(X)
  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (length(Y) != nrow(X) || anyNA(Y)) {
    stop("'Y' must contain one non-missing label per row of 'X'.")
  }
  Y <- droplevels(factor(Y))
  n_classes <- nlevels(Y)
  if (n_classes < 2L) {
    stop("'Y' must contain at least two classes.")
  }

  reg <- .check_finite_scalar(reg, "reg", lower = 0)
  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  max_ncomp <- min(ncol(X), nrow(X) - 1L, n_classes - 1L)
  if (!is.null(ncomp)) {
    ncomp <- .check_whole_number(ncomp, "ncomp", lower = 1L, upper = max_ncomp)
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  n <- nrow(Xp)
  global_mean <- colMeans(Xp)
  centered <- sweep(Xp, 2L, global_mean, FUN = "-")
  total_factor <- t(centered) / sqrt(n)
  total_svd <- svd(total_factor, nu = min(dim(total_factor)), nv = 0L)
  total_cutoff <- tol * max(total_svd$d)
  keep_total <- total_svd$d > total_cutoff
  if (!any(keep_total)) {
    stop("The preprocessed data have zero numerical total-scatter rank.")
  }
  total_vectors <- total_svd$u[, keep_total, drop = FALSE]
  total_values <- total_svd$d[keep_total]

  class_counts <- tabulate(Y, nbins = n_classes)
  class_means <- rowsum(Xp, Y, reorder = TRUE) / class_counts
  mean_offsets <- sweep(class_means, 2L, global_mean, FUN = "-")
  between_factor <- sweep(
    t(mean_offsets), 2L, sqrt(class_counts / n), FUN = "*"
  )

  ridge <- reg * max(total_values^2)
  regularized_values <- sqrt(total_values^2 + ridge)
  reduced_between <- crossprod(total_vectors, between_factor)
  reduced_between <- sweep(
    reduced_between, 1L, regularized_values, FUN = "/"
  )
  between_svd <- svd(
    reduced_between,
    nu = min(dim(reduced_between)),
    nv = 0L
  )
  between_cutoff <- tol * max(between_svd$d)
  keep_between <- between_svd$d > between_cutoff
  between_rank <- sum(keep_between)
  if (between_rank == 0L) {
    stop("The class means have zero numerical between-class scatter.")
  }
  if (!is.null(ncomp) && ncomp > between_rank) {
    stop(sprintf(
      "'ncomp' (%d) exceeds the numerical between-class rank (%d).",
      ncomp, between_rank
    ))
  }
  if (is.null(ncomp)) ncomp <- between_rank

  take <- which(keep_between)[seq_len(ncomp)]
  whitened_total <- sweep(
    total_vectors, 2L, regularized_values, FUN = "/"
  )
  discriminant_basis <- whitened_total %*%
    between_svd$u[, take, drop = FALSE]

  decomposition <- qr(discriminant_basis, LAPACK = TRUE)
  vectors <- qr.Q(decomposition, complete = FALSE)[, seq_len(ncomp), drop = FALSE]
  signs <- vapply(seq_len(ncomp), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")

  component_names <- paste0("FastOLDA", seq_len(ncomp))
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  score_sdev <- apply(scores, 2L, stats::sd)
  names(score_sdev) <- component_names

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = Y,
    classes = "fast_olda",
    eigenvalues = between_svd$d[take]^2,
    ncomp = ncomp,
    total_rank = sum(keep_total),
    between_rank = between_rank,
    reg = reg,
    ridge = ridge,
    tol = tol,
    algorithm = "direct_svd_qr"
  )
}
