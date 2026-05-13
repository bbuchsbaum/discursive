#' Shrinkage Discriminant Analysis
#'
#' Provides shrinkage-based discriminant analysis using [corpcor::sda()]. The
#' method regularises (\emph{shrinks}) the discriminant directions for
#' high-dimensional data, improving robustness and reducing overfitting.
#'
#' @param X A numeric matrix of size `n x d`.
#' @param y A factor or numeric vector of length `n` of class labels. If
#'   numeric, it is coerced to a factor.
#' @param preproc Preprocessing from `multivarious`. Defaults to
#'   [multivarious::center()].
#' @param lambda Regularisation parameter for [corpcor::sda()].
#' @param lambda.var Variance shrinkage parameter for [corpcor::sda()].
#' @param lambda.freqs Class-frequency shrinkage parameter for [corpcor::sda()].
#' @param diagonal Logical. If `TRUE`, restrict the covariance to diagonal.
#'   Default `FALSE`.
#' @param verbose Logical. If `TRUE`, print progress from [corpcor::sda()].
#'   Default `FALSE`.
#'
#' @return A `discriminant_projector` (subclass `"shrinkage_da"`).
#'
#' @details
#' Preprocesses `X`, calls [corpcor::sda()], extracts the discriminant
#' directions `beta` and intercept `alpha`, projects the preprocessed data,
#' and wraps the result in a `discriminant_projector`.
#'
#' @seealso [corpcor::sda()], [multivarious::discriminant_projector()]
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris[, 5]
#' res <- shrinkage_da(X, y, lambda = 0.5)
#' print(res)
#' }
#' @export
shrinkage_da <- function(X, y,
                         preproc = multivarious::center(),
                         lambda,
                         lambda.var,
                         lambda.freqs,
                         diagonal = FALSE,
                         verbose = FALSE) {

  cl <- match.call()
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)

  ret <- corpcor::sda(Xp, y, lambda, lambda.var, lambda.freqs, diagonal, verbose)

  scores <- t(tcrossprod(ret$beta, Xp))

  multivarious::discriminant_projector(
    v       = t(ret$beta),
    s       = scores,
    sdev    = apply(scores, 2, sd),
    preproc = procres,
    labels  = as.character(y),
    alpha   = ret$alpha,
    method  = "shrinkage_da",
    call    = cl,
    classes = "shrinkage_da"
  )
}

#' Project data using a `shrinkage_da` projector
#'
#' Projects new data into the shrinkage discriminant space defined by a
#' `discriminant_projector` of class `"shrinkage_da"`.
#'
#' @param x A `discriminant_projector` of class `"shrinkage_da"`.
#' @param new_data A numeric matrix (or vector) with the same number of
#'   columns as the original data.
#' @param ... Unused.
#'
#' @return A numeric matrix of projected scores.
#' @export
project.shrinkage_da <- function(x, new_data, ...) {
  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), shape(x)[1])
    new_data <- matrix(new_data, byrow = TRUE, ncol = length(new_data))
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, values = nrow(coefficients(x)))

  nd_proc <- reprocess(x, new_data)
  out <- nd_proc %*% coefficients(x)

  if (!is.null(x$alpha)) {
    out <- out + x$alpha
  }
  out
}

#' Partially project data using a `shrinkage_da` projector
#'
#' Projects new data into the shrinkage discriminant space using a subset of
#' columns. A ridge-like scaling is applied: scores are scaled by
#' `nrow(comp) / length(colind)` to adjust for the smaller subset.
#'
#' @param x A `discriminant_projector` of class `"shrinkage_da"`.
#' @param new_data A numeric matrix (or vector) representing partial features.
#' @param colind Indices of the columns to use.
#' @param ... Unused.
#'
#' @return A numeric matrix of projected scores scaled to account for partial
#'   columns.
#' @export
partial_project.shrinkage_da <- function(x, new_data, colind, ...) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, byrow = TRUE, ncol = length(new_data))
  } else if (is.vector(new_data) && length(colind) == 1) {
    new_data <- matrix(new_data, ncol = 1)
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, length(colind))

  comp <- coefficients(x)
  nd_proc <- reprocess(x, new_data, colind)

  out <- nd_proc %*% comp[colind, ]
  out <- out * nrow(comp) / length(colind)

  if (!is.null(x$alpha)) {
    out <- out + x$alpha
  }
  out
}
