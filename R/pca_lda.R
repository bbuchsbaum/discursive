#' Principal Components Followed by Fisher Linear Discriminants
#'
#' Fits a rank-aware PCA plus linear discriminant analysis model, commonly
#' called the Fisherfaces construction. PCA first restricts the predictors to
#' a stable subspace; Fisher LDA is then solved there using the class-size
#' weighted between-class scatter and the within-class scatter.
#'
#' @param X Numeric matrix or data frame with samples in rows and features in
#'   columns.
#' @param Y Class labels with length equal to `nrow(X)`.
#' @param preproc A fitted or unfitted preprocessing specification from
#'   \pkg{multivarious}. The default centers the predictors.
#' @param dp Optional positive integer giving the PCA dimension. The default,
#'   `NULL`, uses at most `nrow(X) - nlevels(Y)` components, as in the
#'   Fisherfaces construction, and never exceeds the numerical PCA rank.
#' @param di Optional positive integer giving the dimension used to whiten the
#'   within-class scatter in PCA space. The default keeps its full positive
#'   numerical rank. If fewer directions are requested, those with the largest
#'   within-class eigenvalues are retained for numerical stability.
#' @param dl Optional positive integer giving the final discriminant dimension.
#'   The default uses the numerical between-class rank, bounded by
#'   `nlevels(Y) - 1` and `di`.
#' @param ncomp Optional compatibility alias for `dl`. Supplying both arguments
#'   is an error unless they are identical.
#' @param tol Relative numerical-rank tolerance in `(0, 1)` for the PCA,
#'   within-class, and between-class eigenspectra.
#'
#' @return A [multivarious::discriminant_projector()] with class `"pca_lda"`.
#'   In addition to the standard projector fields, it contains:
#'   \describe{
#'     \item{eigenvalues}{The Fisher generalized eigenvalues of the returned
#'       discriminant directions.}
#'     \item{dp, di, dl}{The fitted PCA, within-whitening, and final
#'       discriminant dimensions.}
#'     \item{pca_rank, within_rank, between_rank}{Numerical ranks diagnosed at
#'       the three decomposition stages.}
#'     \item{pca_singular_values}{Singular values of the retained PCA axes.}
#'     \item{within_eigenvalues}{Within-class eigenvalues used for whitening.}
#'     \item{pca_center}{The mean used to estimate the PCA basis.}
#'     \item{tol}{The relative rank tolerance used by the fit.}
#'   }
#'
#' @details
#' Let \eqn{P} contain the retained principal axes. In PCA coordinates the
#' method forms the normalized within- and between-class scatters \eqn{S_w}
#' and \eqn{S_b}. Class contributions to \eqn{S_b} are weighted by their sample
#' counts. After whitening the positive part of \eqn{S_w}, it solves the
#' symmetric eigenproblem
#' \deqn{S_b w = \lambda S_w w}
#' and maps the leading solutions back through \eqn{P}. For the full retained
#' within-class rank, the returned directions \eqn{V} satisfy
#' \eqn{V^T S_w V = I} and diagonalize the corresponding between-class scatter.
#'
#' The PCA basis is always estimated from centered preprocessed data. Projected
#' scores still follow `preproc`: with the default centering they are centered,
#' while an explicit [multivarious::pass()] preserves the input origin. The
#' fitted `pca_center` is returned to make this distinction explicit.
#'
#' Directions in the numerical null space of the within-class scatter are
#' discarded rather than inverted. When `dp` is larger than the available
#' within-class rank, `di = NULL` selects only its positive part.
#'
#' @references
#' Belhumeur, P. N., Hespanha, J. P., and Kriegman, D. J. (1997). Eigenfaces
#' vs. Fisherfaces: Recognition Using Class Specific Linear Projection. *IEEE
#' Transactions on Pattern Analysis and Machine Intelligence*, 19(7), 711--720.
#'
#' @seealso [MASS::lda()], [multivarious::pca()],
#'   [multivarious::discriminant_projector()]
#'
#' @importFrom multivarious center
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' fit <- pca_lda(X, iris$Species, dl = 2)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
pca_lda <- function(X,
                    Y,
                    preproc = center(),
                    dp = NULL,
                    di = NULL,
                    dl = NULL,
                    tol = 1e-6,
                    ncomp = NULL) {
  if (!is.null(ncomp)) {
    if (!is.null(dl) &&
        !isTRUE(all.equal(dl, ncomp, check.attributes = FALSE))) {
      stop("'dl' and its compatibility alias 'ncomp' must have the same value when both are supplied.")
    }
    dl <- ncomp
  }
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

  Y <- droplevels(as.factor(Y))
  n_classes <- nlevels(Y)
  if (n_classes < 2L) {
    stop("'Y' must contain at least two classes.")
  }
  within_df <- nrow(X) - n_classes
  if (within_df < 1L) {
    stop("PCA-LDA requires at least one within-class degree of freedom.")
  }

  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  dp_supplied <- !is.null(dp)
  di_supplied <- !is.null(di)
  dl_supplied <- !is.null(dl)
  max_pca_dimension <- min(ncol(X), nrow(X) - 1L)
  if (dp_supplied) {
    dp <- .check_whole_number(
      dp, "dp", lower = 1L, upper = max_pca_dimension
    )
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  n <- nrow(Xp)
  pca_center <- colMeans(Xp)
  centered <- sweep(Xp, 2L, pca_center, FUN = "-")
  pca_svd <- svd(centered, nu = 0L, nv = min(dim(centered)))
  pca_cutoff <- tol * max(pca_svd$d)
  pca_rank <- sum(pca_svd$d > pca_cutoff)
  if (pca_rank == 0L) {
    stop("The preprocessed data have zero numerical PCA rank.")
  }

  if (!dp_supplied) {
    dp <- min(ncol(X), within_df, pca_rank)
  } else if (dp > pca_rank) {
    stop(sprintf(
      "'dp' (%d) exceeds the numerical PCA rank (%d).", dp, pca_rank
    ))
  }
  pca_vectors <- pca_svd$v[, seq_len(dp), drop = FALSE]
  pca_scores <- centered %*% pca_vectors

  class_counts <- tabulate(Y, nbins = n_classes)
  class_means <- rowsum(pca_scores, Y, reorder = TRUE) / class_counts
  within_residuals <- pca_scores - class_means[as.integer(Y), , drop = FALSE]
  within_scatter <- crossprod(within_residuals) / n

  mean_offsets <- sweep(class_means, 2L, colMeans(pca_scores), FUN = "-")
  weighted_offsets <- sweep(
    mean_offsets, 1L, sqrt(class_counts / n), FUN = "*"
  )
  between_scatter <- crossprod(weighted_offsets)

  within_decomp <- eigen(
    0.5 * (within_scatter + t(within_scatter)), symmetric = TRUE
  )
  within_max <- max(within_decomp$values)
  if (!is.finite(within_max) || within_max <= 0) {
    stop("The PCA scores have zero positive within-class scatter rank.")
  }
  within_cutoff <- tol * within_max
  within_rank <- sum(within_decomp$values > within_cutoff)

  if (!di_supplied) {
    di <- within_rank
  } else {
    di <- .check_whole_number(di, "di", lower = 1L, upper = dp)
    if (di > within_rank) {
      stop(sprintf(
        "'di' (%d) exceeds the numerical within-class rank (%d).",
        di, within_rank
      ))
    }
  }

  within_values <- within_decomp$values[seq_len(di)]
  whitening <- sweep(
    within_decomp$vectors[, seq_len(di), drop = FALSE],
    2L,
    sqrt(within_values),
    FUN = "/"
  )
  whitened_between <- crossprod(
    whitening, between_scatter %*% whitening
  )
  whitened_between <- 0.5 * (whitened_between + t(whitened_between))
  between_decomp <- eigen(whitened_between, symmetric = TRUE)
  between_max <- max(between_decomp$values)
  if (!is.finite(between_max) || between_max <= 0) {
    stop("The class means have zero positive between-class scatter rank.")
  }
  between_cutoff <- tol * between_max
  between_rank <- sum(between_decomp$values > between_cutoff)

  max_discriminant_dimension <- min(di, n_classes - 1L)
  if (!dl_supplied) {
    dl <- min(max_discriminant_dimension, between_rank)
  } else {
    dl <- .check_whole_number(
      dl, "dl", lower = 1L, upper = max_discriminant_dimension
    )
    if (dl > between_rank) {
      stop(sprintf(
        "'dl' (%d) exceeds the numerical between-class rank (%d).",
        dl, between_rank
      ))
    }
  }

  discriminant_pca <- whitening %*%
    between_decomp$vectors[, seq_len(dl), drop = FALSE]
  vectors <- pca_vectors %*% discriminant_pca

  # Resolve arbitrary decomposition signs for reproducible loadings and scores.
  signs <- vapply(seq_len(dl), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")

  component_names <- paste0("PCALDA", seq_len(dl))
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  centered_scores <- sweep(scores, 2L, colMeans(scores), FUN = "-")
  score_sdev <- sqrt(colSums(centered_scores^2) / (n - 1L))
  names(score_sdev) <- component_names

  eigenvalues <- between_decomp$values[seq_len(dl)]
  names(eigenvalues) <- component_names
  names(within_values) <- paste0("Within", seq_len(di))

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = Y,
    classes = "pca_lda",
    eigenvalues = eigenvalues,
    dp = dp,
    di = di,
    dl = dl,
    pca_rank = pca_rank,
    within_rank = within_rank,
    between_rank = between_rank,
    pca_singular_values = pca_svd$d[seq_len(dp)],
    within_eigenvalues = within_values,
    pca_center = pca_center,
    tol = tol
  )
}
