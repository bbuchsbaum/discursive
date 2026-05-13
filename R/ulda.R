#' Uncorrelated Linear Discriminant Analysis (ULDA)
#'
#' Performs Uncorrelated Linear Discriminant Analysis. ULDA seeks linear
#' combinations of features that maximise class separation while ensuring the
#' resulting discriminant axes are mutually uncorrelated.
#'
#' @param X A numeric matrix of size `n x d` (rows = samples, columns = features).
#' @param y A factor or numeric vector of length `n` of class labels. If
#'   numeric, it is coerced to a factor.
#' @param ncomp Integer. Number of discriminant components to retain. If
#'   `NULL` (default), all components with positive singular values are kept.
#' @param preproc A preprocessing function from `multivarious`. Defaults to
#'   [multivarious::center()].
#' @param mu Regularisation parameter (currently unused, kept for
#'   extensibility). Default 0.
#' @param tol Tolerance — singular values smaller than `tol` are discarded.
#'   Default `1e-6`.
#'
#' @return A `discriminant_projector` object (subclass `"ulda"`).
#'
#' @details
#' The procedure: preprocess `X`; compute class probabilities, class means and
#' the global mean; form between-class scatter `Hb`; centre data around the
#' global mean to form `Ht`; SVD `Ht`, then SVD a transformed `Hb` to find
#' discriminant directions; combine the transformations and project.
#'
#' @seealso [multivarious::prep()], [RSpectra::eigs_sym()],
#'   [multivarious::discriminant_projector()]
#'
#' @importFrom multivarious prep
#' @export
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris[, 5]
#' res <- ulda(X, y)
#' }
ulda <- function(X, y, ncomp = NULL, preproc = multivarious::center(),
                 mu = 0, tol = 1e-6) {
  cl <- match.call()
  if (!is.matrix(X)) X <- as.matrix(X)
  if (nrow(X) != length(y)) {
    stop("'y' must have the same length as the number of rows in 'X'.")
  }
  y <- as.factor(y)

  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)

  nc <- nlevels(y)
  cprobs <- table(y) / length(y)
  cmeans <- group_means(y, Xp)
  gmean  <- colMeans(Xp)

  Hb <- do.call(cbind, lapply(seq_len(nc), function(i) {
    sqrt(cprobs[i]) * (cmeans[i, ] - gmean)
  }))

  Ht <- t(sweep(Xp, 2, gmean, "-"))

  svd_ht <- svd(Ht)
  keep   <- which(svd_ht$d > tol)

  Ut_scaled <- t(sweep(svd_ht$u[, keep, drop = FALSE], 2, svd_ht$d[keep], "/"))
  B <- Ut_scaled %*% Hb

  svd_B  <- svd(B)
  keep_b <- which(svd_B$d > tol)

  U_scaled <- sweep(svd_ht$u[, keep, drop = FALSE], 2, svd_ht$d[keep], "/")
  vecs <- U_scaled %*% svd_B$u[, keep_b, drop = FALSE]

  if (!is.null(ncomp)) {
    ncomp <- as.integer(ncomp)
    if (ncomp < 1 || ncomp > ncol(vecs)) {
      stop(sprintf("'ncomp' must be between 1 and %d.", ncol(vecs)))
    }
    vecs <- vecs[, seq_len(ncomp), drop = FALSE]
  }

  s <- Xp %*% vecs

  multivarious::discriminant_projector(
    v       = vecs,
    s       = s,
    sdev    = apply(s, 2, sd),
    preproc = procres,
    labels  = y,
    method  = "ulda",
    call    = cl,
    classes = "ulda"
  )
}
