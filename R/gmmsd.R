#' Generalized Multiple Maximum Scatter Difference
#'
#' Fits the generalized multiple maximum scatter difference (GMMSD) feature
#' extractor. GMMSD maximizes a between-minus-within scatter criterion in the
#' numerical row space of the centered, preprocessed data.
#'
#' @param X Numeric matrix or data frame with samples in rows and features in
#'   columns.
#' @param y Class labels with length equal to \code{nrow(X)}.
#' @param c Finite nonnegative scalar multiplying the within-class scatter.
#'   Larger values penalize within-class variation more strongly.
#' @param dim Positive integer giving the number of projection dimensions.
#'   It cannot exceed the numerical rank of the centered, preprocessed data.
#' @param ncomp Optional compatibility alias for `dim`. Supplying both
#'   arguments is an error unless they have the same value.
#' @param preproc A fitted or unfitted preprocessing specification from
#'   \pkg{multivarious}. The default centers the predictors.
#' @param tol Relative QR rank tolerance in \code{(0, 1)}. A diagonal direction
#'   is retained when its absolute pivoted-QR diagonal exceeds \code{tol} times
#'   the largest absolute diagonal.
#'
#' @return A [multivarious::discriminant_projector()] with class \code{"gmmsd"}.
#'   In addition to the standard projector fields, it contains:
#'   \describe{
#'     \item{eigenvalues}{Scatter-difference eigenvalues for the returned
#'       directions.}
#'     \item{criterion_eigenvalues}{The complete reduced criterion spectrum.}
#'     \item{dim}{The fitted projection dimension.}
#'     \item{optimization_rank}{Numerical rank of the internally centered,
#'       preprocessed data.}
#'     \item{qr_diagonal, rank_threshold}{Absolute pivoted-QR diagonals and the
#'       threshold used to select the numerical row space after global scale
#'       normalization.}
#'     \item{c, tol}{The fitted balance parameter and rank tolerance.}
#'     \item{data_scale}{Global scale removed for the numerical solve. Returned
#'       eigenvalues are restored to the original preprocessed-data units.}
#'     \item{criterion_scale_normalized}{Largest absolute entry of the reduced
#'       scatter-difference matrix in normalized coordinates.}
#'     \item{criterion_center}{The center removed only while estimating the
#'       GMMSD directions. Scores continue to follow \code{preproc}.}
#'   }
#'
#' @details
#' Let \eqn{S_b} and \eqn{S_w} denote the unnormalized between-class and
#' within-class scatter matrices of the preprocessed data. If \eqn{Q} is an
#' orthonormal basis for its numerical centered row space, GMMSD solves
#' \deqn{\max_{U^T U=I} \operatorname{tr}
#'       \{U^T Q^T(S_b-cS_w)QU\}}
#' and returns \eqn{V=QU}. Thus the returned loadings are Euclidean orthonormal
#' and satisfy the scatter-difference eigenproblem in the retained row space.
#'
#' The implementation never constructs feature-by-feature scatter matrices.
#' Class scatters are formed after projection into the at-most
#' \code{min(nrow(X) - 1, ncol(X))} dimensional QR space. This preserves the
#' purpose of the QR formulation for high-dimensional data and excludes
#' arbitrary numerical-null directions.
#'
#' Directions are estimated after an internal centering step even when
#' \code{preproc} does not center. This makes the fitted subspace translation
#' invariant. Training and future scores are still computed from data
#' transformed only by \code{preproc}, so an explicit
#' [multivarious::pass()] preserves the requested score origin.
#'
#' @references
#' Zheng, N., Qi, L., and Guan, L. (2014). Generalized multiple maximum scatter
#' difference feature extraction using QR decomposition. *Journal of Visual
#' Communication and Image Representation*, 25(6), 1460--1471.
#' \doi{10.1016/j.jvcir.2014.04.009}
#'
#' @seealso [between_class_scatter()], [within_class_scatter()],
#'   [multivarious::discriminant_projector()]
#'
#' @importFrom multivarious center
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' fit <- gmmsd(X, iris$Species, c = 1, dim = 2)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
gmmsd <- function(X,
                  y,
                  c = 1,
                  dim = 2,
                  preproc = multivarious::center(),
                  tol = 1e-8,
                  ncomp = NULL) {
  dim_supplied <- !missing(dim)
  if (!is.null(ncomp)) {
    if (dim_supplied && !isTRUE(all.equal(dim, ncomp, check.attributes = FALSE))) {
      stop("'dim' and its compatibility alias 'ncomp' must have the same value when both are supplied.")
    }
    dim <- ncomp
  }
  X <- as.matrix(X)
  if (!is.numeric(X) || length(base::dim(X)) != 2L ||
      nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (length(y) != nrow(X) || anyNA(y)) {
    stop("'y' must contain one non-missing label per row of 'X'.")
  }
  if ((!is.atomic(y) && !is.factor(y)) || !is.null(base::dim(y))) {
    stop("'y' must be an atomic vector or factor.")
  }

  y <- droplevels(as.factor(y))
  if (nlevels(y) < 2L) {
    stop("'y' must contain at least two classes.")
  }

  c <- .check_finite_scalar(c, "c", lower = 0)
  dim <- .check_whole_number(dim, "dim", lower = 1L)
  tol <- .check_finite_scalar(
    tol,
    "tol",
    lower = 0,
    upper = 1,
    lower_open = TRUE,
    upper_open = TRUE
  )

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || nrow(Xp) != nrow(X) || ncol(Xp) != ncol(X) ||
      any(!is.finite(Xp))) {
    stop(
      "'preproc' must produce a finite numeric matrix with the same dimensions as 'X'."
    )
  }

  # Center relative to an observed row first. This avoids summing a large
  # common offset and keeps the estimated scatter subspace translation stable.
  shifted <- sweep(Xp, 2L, Xp[1L, ], FUN = "-")
  if (any(!is.finite(shifted))) {
    stop("Pairwise differences in preprocessed 'X' overflowed; rescale 'X'.")
  }
  shifted_center <- colMeans(shifted)
  centered <- sweep(shifted, 2L, shifted_center, FUN = "-")
  criterion_center <- Xp[1L, ] + shifted_center
  data_scale <- max(abs(centered))
  if (!is.finite(data_scale) || data_scale == 0) {
    stop("The preprocessed data have zero numerical rank.")
  }
  scatter_scale <- data_scale * data_scale
  if (!is.finite(scatter_scale) || scatter_scale == 0) {
    stop(
      "The scatter scale is outside the representable numeric range; rescale 'X'."
    )
  }
  centered_scaled <- centered / data_scale

  decomposition <- qr(t(centered_scaled), LAPACK = TRUE)
  triangular <- qr.R(decomposition, complete = FALSE)
  qr_diagonal <- abs(diag(triangular))
  qr_scale <- if (length(qr_diagonal)) max(qr_diagonal) else 0
  if (!is.finite(qr_scale) || qr_scale == 0) {
    stop("The preprocessed data have zero numerical rank.")
  }
  rank_threshold <- tol * qr_scale
  retained <- which(qr_diagonal > rank_threshold)
  optimization_rank <- length(retained)
  if (optimization_rank == 0L) {
    stop("The preprocessed data have zero numerical rank at the requested 'tol'.")
  }
  if (dim > optimization_rank) {
    stop(sprintf(
      "'dim' (%d) exceeds the numerical rank of the preprocessed data (%d).",
      dim,
      optimization_rank
    ))
  }

  full_basis <- qr.Q(decomposition, complete = FALSE)
  row_basis <- full_basis[, retained, drop = FALSE]
  reduced_scores <- centered_scaled %*% row_basis

  statistics <- .scatter_group_statistics(
    reduced_scores,
    y,
    compute_within = TRUE,
    compute_between = TRUE
  )
  criterion <- statistics$between - c * statistics$within
  criterion <- 0.5 * (criterion + t(criterion))
  if (any(!is.finite(criterion))) {
    stop("The reduced scatter-difference matrix overflowed; rescale 'X'.")
  }
  criterion_scale_normalized <- max(abs(criterion))
  if (criterion_scale_normalized == 0) {
    stop("The reduced scatter-difference matrix has zero numerical scale.")
  }

  criterion_decomposition <- eigen(criterion, symmetric = TRUE)
  take <- seq_len(dim)
  vectors <- row_basis %*%
    criterion_decomposition$vectors[, take, drop = FALSE]

  # Resolve arbitrary eigenvector signs so repeated fits have stable loadings.
  signs <- vapply(take, function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")

  component_names <- paste0("GMMSD", take)
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  centered_scores <- sweep(scores, 2L, colMeans(scores), FUN = "-")
  score_sdev <- sqrt(colSums(centered_scores^2) / (nrow(X) - 1L))
  names(score_sdev) <- component_names

  criterion_eigenvalues <- criterion_decomposition$values * scatter_scale
  if (any(!is.finite(criterion_eigenvalues))) {
    stop("The scatter-difference eigenvalues overflowed; rescale 'X'.")
  }
  eigenvalues <- criterion_eigenvalues[take]
  names(eigenvalues) <- component_names

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = y,
    classes = "gmmsd",
    eigenvalues = eigenvalues,
    criterion_eigenvalues = criterion_eigenvalues,
    dim = dim,
    optimization_rank = optimization_rank,
    qr_diagonal = qr_diagonal,
    rank_threshold = rank_threshold,
    criterion_scale_normalized = criterion_scale_normalized,
    criterion_center = criterion_center,
    data_scale = data_scale,
    c = c,
    tol = tol
  )
}
