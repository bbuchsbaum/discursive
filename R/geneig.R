#' Symmetric Generalized Eigenvalue Decomposition
#'
#' Solves \eqn{A v = \lambda B v} for symmetric `A` and positive-semidefinite
#' `B`, returning the algebraically largest eigenpairs in the positive
#' numerical range of `B`.
#'
#' @param A Symmetric finite numeric left-hand-side matrix.
#' @param B Symmetric finite positive-semidefinite right-hand-side matrix.
#' @param ncomp Positive integer number of eigenpairs, not exceeding the
#'   positive numerical rank of `B`.
#' @param method `"robust"` or its compatibility alias `"sdiag"` uses an
#'   explicit symmetric whitening solve. `"lapack"` uses [geigen::geigen()]
#'   and requires positive-definite `B`.
#' @param tol Relative tolerance in `(0, 1)` used to determine the positive
#'   numerical rank of `B`.
#'
#' @return A [multivarious::projector()] with class `"geneig"`, coefficients
#'   in `v`, generalized eigenvalues in `values`, and rank/method diagnostics.
#'
#' @details
#' The robust path diagonalizes `B`, restricts the problem to its positive
#' eigenspace, and solves the symmetric whitened problem. The returned vectors
#' satisfy \eqn{V^T B V = I} and are ordered by algebraic eigenvalue, largest
#' first. This avoids applying a symmetric eigensolver to the generally
#' nonsymmetric product `solve(B, A)`.
#'
#' @export
#'
#' @examples
#' A <- matrix(c(5, 1, 1, 2), 2)
#' B <- diag(c(1, 2))
#' fit <- geneig(A, B, ncomp = 2)
#' crossprod(fit$v, B %*% fit$v)
geneig <- function(A,
                   B,
                   ncomp = 2,
                   method = c("robust", "sdiag", "lapack"),
                   tol = 1e-8) {
  method <- match.arg(method)
  A <- as.matrix(A)
  B <- as.matrix(B)
  if (!is.numeric(A) || !is.numeric(B)) {
    stop("'A' and 'B' must be numeric matrices.")
  }
  if (nrow(A) != ncol(A) || nrow(B) != ncol(B) || any(dim(A) != dim(B))) {
    stop("'A' and 'B' must be square matrices with the same dimensions.")
  }
  if (any(!is.finite(A)) || any(!is.finite(B))) {
    stop("'A' and 'B' must contain only finite values.")
  }
  if (!isSymmetric(A) || !isSymmetric(B)) {
    stop("'A' and 'B' must be symmetric.")
  }
  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )

  B_decomp <- eigen(B, symmetric = TRUE)
  B_scale <- max(abs(B_decomp$values))
  if (!is.finite(B_scale) || B_scale == 0) {
    stop("'B' must have positive numerical rank.")
  }
  if (min(B_decomp$values) < -tol * B_scale) {
    stop("'B' must be positive semidefinite.")
  }
  keep <- B_decomp$values > tol * B_scale
  rank_B <- sum(keep)
  if (rank_B == 0L) {
    stop("'B' must have positive numerical rank.")
  }
  ncomp <- .check_whole_number(ncomp, "ncomp", lower = 1L, upper = rank_B)

  if (method == "lapack") {
    if (rank_B < nrow(B)) {
      stop("method = 'lapack' requires positive-definite 'B'.")
    }
    if (!requireNamespace("geigen", quietly = TRUE)) {
      stop("method = 'lapack' requires the 'geigen' package.")
    }
    result <- geigen::geigen(A, B, symmetric = TRUE)
    values <- as.numeric(result$values)
    vectors <- as.matrix(result$vectors)
    ordering <- order(values, decreasing = TRUE)
    take <- ordering[seq_len(ncomp)]
    values <- values[take]
    vectors <- vectors[, take, drop = FALSE]
    norms <- sqrt(colSums(vectors * (B %*% vectors)))
    vectors <- sweep(vectors, 2L, norms, FUN = "/")
  } else {
    positive_vectors <- B_decomp$vectors[, keep, drop = FALSE]
    positive_values <- B_decomp$values[keep]
    whitener <- sweep(
      positive_vectors, 2L, sqrt(positive_values), FUN = "/"
    )
    whitened_A <- crossprod(whitener, A %*% whitener)
    whitened_A <- 0.5 * (whitened_A + t(whitened_A))
    A_decomp <- eigen(whitened_A, symmetric = TRUE)
    values <- A_decomp$values[seq_len(ncomp)]
    vectors <- whitener %*%
      A_decomp$vectors[, seq_len(ncomp), drop = FALSE]
  }

  signs <- vapply(seq_len(ncomp), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")
  component_names <- paste0("GE", seq_len(ncomp))
  rownames(vectors) <- rownames(A)
  colnames(vectors) <- component_names
  names(values) <- component_names

  preproc_obj <- multivarious::fit(
    multivarious::pass(), matrix(0, nrow = 1L, ncol = nrow(A))
  )
  multivarious::projector(
    v = vectors,
    preproc = preproc_obj,
    classes = "geneig",
    values = values,
    rank_B = rank_B,
    method = method,
    tol = tol
  )
}
