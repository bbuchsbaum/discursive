#' Uncorrelated Linear Discriminant Analysis
#'
#' Fits the SVD-based uncorrelated linear discriminant analysis (ULDA)
#' solution of Ye (2005). The returned discriminant scores are mutually
#' uncorrelated with respect to the empirical total scatter matrix.
#'
#' @param X Numeric matrix or data frame with samples in rows and features in
#'   columns.
#' @param Y Class labels with length equal to `nrow(X)`.
#' @param preproc A fitted or unfitted preprocessing specification from
#'   \pkg{multivarious}. The default centers the predictors.
#' @param mu Nonnegative compatibility parameter. The SVD solution is
#'   independent of `mu`, so this value is validated and stored but does not
#'   alter the fitted projection.
#' @param tol Relative numerical-rank tolerance in `(0, 1)`. A singular value
#'   is retained when it is greater than `tol` times the largest singular value
#'   of the same matrix.
#' @param ncomp Optional positive integer giving the number of discriminant
#'   components. The default, `NULL`, returns the numerical rank of the
#'   between-class scatter, which is at most the number of classes minus one.
#'
#' @return A [multivarious::discriminant_projector()] with class `"ulda"`.
#'   In addition to the standard projector fields, it contains:
#'   \describe{
#'     \item{eigenvalues}{The generalized eigenvalues associated with the
#'       returned discriminant directions.}
#'     \item{ncomp}{The number of returned components.}
#'     \item{total_rank}{The numerical rank of the total scatter matrix.}
#'     \item{tol}{The relative rank tolerance used by the fit.}
#'     \item{mu}{The validated compatibility value supplied to `mu`.}
#'   }
#'
#' @details
#' Let \eqn{S_t} and \eqn{S_b} denote the total and between-class scatter
#' matrices, each normalized by the sample size. ULDA computes a basis \eqn{G}
#' that simultaneously diagonalizes the scatter matrices and satisfies
#' \deqn{G^T S_t G = I.}
#' Consequently, the fitted score columns are uncorrelated. The implementation
#' follows Algorithm 1 of Ye (2005): it whitens a reduced SVD of the centered
#' data matrix and then rotates that basis using an SVD of the whitened class
#' centroids. Only left singular vectors are computed because right singular
#' vectors are not needed.
#'
#' `mu` is retained for compatibility with the previous function signature.
#' The cited derivation shows that the optimal ULDA transformation is
#' independent of this parameter; it is not a ridge penalty in this
#' implementation.
#'
#' @references
#' Ye, J. (2005). Characterization of a Family of Algorithms for Generalized
#' Discriminant Analysis on Undersampled Problems. *Journal of Machine Learning
#' Research*, 6, 483--502.
#'
#' @seealso [olda()], [multivarious::discriminant_projector()]
#'
#' @importFrom multivarious center fit transform
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' fit <- ulda(X, iris$Species, ncomp = 2)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
ulda <- function(X,
                 Y,
                 preproc = center(),
                 mu = 0,
                 tol = 1e-6,
                 ncomp = NULL) {
  cl <- match.call()
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

  mu <- .check_finite_scalar(mu, "mu", lower = 0)
  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  max_ncomp <- min(ncol(X), nrow(X) - 1L, n_classes - 1L)
  if (!is.null(ncomp)) {
    ncomp <- .check_whole_number(
      ncomp, "ncomp", lower = 1L, upper = max_ncomp
    )
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  n <- nrow(Xp)
  global_mean <- colMeans(Xp)
  class_counts <- tabulate(Y, nbins = n_classes)
  class_means <- rowsum(Xp, Y, reorder = TRUE) / class_counts

  # Hb %*% t(Hb) is the between-class scatter normalized by n.
  mean_offsets <- sweep(class_means, 2L, global_mean, FUN = "-")
  Hb <- sweep(
    t(mean_offsets), 2L, sqrt(class_counts / n), FUN = "*"
  )

  # Ht %*% t(Ht) is the total scatter normalized by n. Keeping the
  # 1 / sqrt(n) factor is necessary for G' St G = I.
  centered <- sweep(Xp, 2L, global_mean, FUN = "-")
  Ht <- t(centered) / sqrt(n)
  svd_total <- svd(Ht, nu = min(dim(Ht)), nv = 0L)
  total_cutoff <- tol * max(svd_total$d)
  keep_total <- svd_total$d > total_cutoff
  if (!any(keep_total)) {
    stop("The preprocessed data have zero numerical total-scatter rank.")
  }

  total_vectors <- svd_total$u[, keep_total, drop = FALSE]
  total_values <- svd_total$d[keep_total]

  # B = Sigma_t^-1 U_t' Hb. sweep() avoids diag(scalar), which is not a
  # one-by-one diagonal matrix in R, and avoids materializing a diagonal matrix.
  B <- crossprod(total_vectors, Hb)
  B <- sweep(B, 1L, total_values, FUN = "/")
  svd_between <- svd(B, nu = min(dim(B)), nv = 0L)
  between_cutoff <- tol * max(svd_between$d)
  keep_between <- svd_between$d > between_cutoff
  effective_rank <- sum(keep_between)
  if (effective_rank == 0L) {
    stop("The class means have zero numerical between-class scatter.")
  }
  if (!is.null(ncomp) && ncomp > effective_rank) {
    stop(sprintf(
      "'ncomp' (%d) exceeds the numerical between-class rank (%d).",
      ncomp, effective_rank
    ))
  }
  if (is.null(ncomp)) {
    ncomp <- effective_rank
  }

  take <- which(keep_between)[seq_len(ncomp)]
  whitened_total <- sweep(total_vectors, 2L, total_values, FUN = "/")
  vectors <- whitened_total %*%
    svd_between$u[, take, drop = FALSE]

  # Resolve arbitrary SVD signs for reproducible coefficients and scores.
  signs <- vapply(seq_len(ncomp), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")

  component_names <- paste0("ULDA", seq_len(ncomp))
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  centered_scores <- sweep(scores, 2L, colMeans(scores), FUN = "-")
  score_sdev <- sqrt(colSums(centered_scores^2) / (n - 1L))
  names(score_sdev) <- component_names

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = Y,
    classes = "ulda",
    method = "ulda",
    call = cl,
    eigenvalues = svd_between$d[take]^2,
    ncomp = ncomp,
    total_rank = sum(keep_total),
    tol = tol,
    mu = mu
  )
}
