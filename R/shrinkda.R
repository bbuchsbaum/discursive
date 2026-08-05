#' Shrinkage Discriminant Analysis
#'
#' This function provides a shrinkage-based discriminant analysis using the
#' \code{\link[sda]{sda}} routine from the \pkg{sda} package.
#' The method can regularize (\emph{shrink}) the discriminant directions for
#' high-dimensional data, potentially improving robustness and reducing overfitting.
#'
#' @param X A numeric matrix of size `n` by `d` (samples by features).
#' @param Y A factor or numeric vector of length \code{n} representing class labels.
#'          If numeric, it will be converted to a factor.
#' @param preproc A pre-processing function or object (from multivarious) applied to \code{X} before shrinkage DA. Defaults to \code{center()}.
#' @param lambda Optional correlation shrinkage intensity in `[0, 1]`. If
#'   omitted, [sda::sda()] estimates it.
#' @param lambda.var Regularization parameter for the variance (covariance) terms,
#'   passed to [sda::sda()]. If omitted, it is estimated.
#' @param lambda.freqs Regularization parameter for class frequency shrinkage,
#'   passed to [sda::sda()]. If omitted, it is estimated.
#' @param diagonal Logical, if \code{TRUE}, the method restricts the covariance matrix
#'   to be diagonal. Defaults to \code{FALSE}.
#' @param verbose Logical, if \code{TRUE}, prints progress messages from \code{\link[sda]{sda}}.
#'   Defaults to \code{FALSE}.
#'
#' @return A \code{\link[multivarious]{discriminant_projector}} object with
#'   classes `"shrinkage_da"` and `"shrinkda"`, containing:
#' \itemize{
#'   \item \code{v} : A `d` by `m` matrix of discriminant directions
#'         (\code{ret\$beta} transposed).
#'   \item \code{s} : An `n` by `m` matrix of intercept-adjusted discriminant
#'         scores for the training data.
#'   \item \code{sdev} : Standard deviations of each column in \code{s}.
#'   \item \code{labels} : The class labels (in character form).
#'   \item \code{alpha} : Intercept or offset term from \code{\link[sda]{sda}} result (if any).
#'   \item \code{preproc} : The pre-processing object used.
#'   \item \code{classes} : Includes `"shrinkage_da"` and `"shrinkda"`.
#' }
#'
#' @details
#' Internally, the function:
#' \enumerate{
#'   \item Preprocesses \code{X} using \code{preproc}.
#'   \item Calls \code{\link[sda]{sda}} with the specified \code{lambda}, \code{lambda.var}, \code{lambda.freqs}, etc.
#'   \item Extracts \code{ret\$beta} (the discriminant directions) and \code{ret\$alpha} (the intercept).
#'   \item Projects the preprocessed data to obtain the score matrix \code{s}.
#'   \item Constructs a \code{discriminant_projector} object with class \code{"shrinkda"}.
#' }
#'
#' @seealso \code{\link[sda]{sda}}, \code{\link[multivarious]{discriminant_projector}}
#'
#' @examples
#' \dontrun{
#' library(multivarious)
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' Y <- iris[, 5]
#'
#' # Perform shrinkage DA with default centering
#' res <- shrinkage_da(X, Y, lambda=0.5)
#'
#' # Inspect the projector
#' print(res)
#'
#' # Project new data
#' # new_data <- ...
#' # projected_scores <- project(res, new_data)
#' }
#' @export
shrinkage_da <- function(X,
                         Y,
                         preproc = center(),
                         lambda,
                         lambda.var,
                         lambda.freqs,
                         diagonal = FALSE,
                         verbose = FALSE) {
  lambda_missing <- missing(lambda)
  lambda_var_missing <- missing(lambda.var)
  lambda_freqs_missing <- missing(lambda.freqs)

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
  if (nlevels(Y) < 2L) {
    stop("'Y' must contain at least two classes.")
  }
  if (!is.logical(diagonal) || length(diagonal) != 1L || is.na(diagonal)) {
    stop("'diagonal' must be TRUE or FALSE.")
  }
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("'verbose' must be TRUE or FALSE.")
  }
  for (parameter in c("lambda", "lambda.var", "lambda.freqs")) {
    is_missing <- switch(
      parameter,
      lambda = lambda_missing,
      lambda.var = lambda_var_missing,
      lambda.freqs = lambda_freqs_missing
    )
    if (!is_missing) {
      value <- get(parameter, inherits = FALSE)
      .check_finite_scalar(value, parameter, lower = 0, upper = 1)
    }
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  arguments <- list(
    Xtrain = Xp,
    L = Y,
    diagonal = diagonal,
    verbose = verbose
  )
  if (!lambda_missing) arguments$lambda <- lambda
  if (!lambda_var_missing) arguments$lambda.var <- lambda.var
  if (!lambda_freqs_missing) arguments$lambda.freqs <- lambda.freqs
  ret <- do.call(sda::sda, arguments)

  vectors <- t(ret$beta)
  scores <- Xp %*% vectors
  if (!is.null(ret$alpha)) {
    scores <- sweep(scores, 2L, ret$alpha, FUN = "+")
  }

  multivarious::discriminant_projector(
    v       = vectors,
    s       = scores,
    sdev    = apply(scores, 2, sd),
    preproc = procres,
    labels  = Y,
    alpha   = ret$alpha,
    regularization = ret$regularization,
    frequencies = ret$freqs,
    classes = c("shrinkage_da", "shrinkda")
  )
}

#' Project data using a shrinkage_da projector
#'
#' This S3 method projects new data into the shrinkage discriminant space defined
#' by a \code{discriminant_projector} of class \code{"shrinkda"}. It applies
#' any stored preprocessing, multiplies by \code{coefficients(x)}
#' (i.e., \code{x\$v}), and adds \code{x\$alpha} if provided.
#'
#' @param x A \code{discriminant_projector} object of class \code{"shrinkda"}.
#' @param new_data A numeric matrix (or vector) with the same number of columns as
#'        the original data (unless partial usage).
#' @param ... Further arguments (not used).
#'
#' @return A numeric matrix of projected scores.
#' @export
project.shrinkage_da <- function(x, new_data, ...) {
  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), multivarious::shape(x)[1])
    new_data <- matrix(new_data, byrow = TRUE, ncol = length(new_data))
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, values = nrow(coefficients(x)))

  # 1) Preprocess the new data
  nd_proc <- multivarious::reprocess(x, new_data)

  # 2) Multiply by x$v
  out <- nd_proc %*% coefficients(x)

  # 3) Add intercept offset if stored
  if (!is.null(x$alpha)) {
    out <- sweep(out, 2L, x$alpha, FUN = "+")
  }

  return(out)
}

#' Partially project data using a shrinkage_da projector
#'
#' This S3 method projects new data into the shrinkage discriminant space,
#' but allows the user to specify a subset of columns from the original data
#' (i.e., partial usage). A ridge-like scaling is also applied:
#' we scale by \code{nrow(comp)/length(colind)} to adjust for the smaller subset.
#'
#' @param x A \code{discriminant_projector} object of class \code{"shrinkda"}.
#' @param new_data A numeric matrix (or vector) representing partial features.
#' @param colind A numeric vector giving the indices of columns in the original
#'        data to use for the partial projection.
#' @param ... Further arguments (not used).
#'
#' @rdname partial_project.shrinkage_da
#' @importFrom multivarious project
#' @export
partial_project <- function(x, ...) {
  UseMethod("partial_project")
}

#' @return A numeric matrix of projected scores, scaled to account for partial columns.
#' @export
partial_project.shrinkage_da <- function(x, new_data, colind, ...) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, byrow = TRUE, ncol = length(new_data))
  } else if (is.vector(new_data) && length(colind) == 1) {
    new_data <- matrix(new_data, ncol = 1)
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, length(colind))

  # 1) Subset the coefficients
  comp <- coefficients(x)

  # 2) Reprocess partial data
  nd_proc <- multivarious::reprocess(x, new_data, colind)

  # 3) Multiply by the corresponding rows in comp
  #    Then scale by nrow(comp)/length(colind)
  out <- nd_proc %*% comp[colind, ]

  out <- out * nrow(comp) / length(colind)

  # 4) Add alpha offset if present
  if (!is.null(x$alpha)) {
    out <- sweep(out, 2L, x$alpha, FUN = "+")
  }

  return(out)
}
