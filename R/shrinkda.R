#' Shrinkage Discriminant Analysis
#'
#' This function provides a shrinkage-based discriminant analysis using the 
#' \code{\link[corpcor]{sda}} routine from the \pkg{corpcor} package. 
#' The method can regularize (\emph{shrink}) the discriminant directions for 
#' high-dimensional data, potentially improving robustness and reducing overfitting.
#'
#' @param X A numeric matrix of size \code{n \times d} (samples by features).
#' @param Y A factor or numeric vector of length \code{n} representing class labels. 
#'          If numeric, it will be converted to a factor.
#' @param preproc A pre-processing function or object from \pkg{multivarious} 
#'   (e.g. \code{\link[multivarious]{center}}, \code{\link[multivarious]{scale}}), 
#'   applied to \code{X} before shrinkage DA. Defaults to \code{\link[multivarious]{center}}.
#' @param lambda Regularization parameter for the \code{\link[corpcor]{sda}} function 
#'   (shrinkage of the discriminant directions). Typically in \eqn{[0, 1]}.
#' @param lambda.var Regularization parameter for the variance (covariance) terms, 
#'   passed to \code{\link[corpcor]{sda}}.
#' @param lambda.freqs Regularization parameter for class frequency shrinkage, 
#'   passed to \code{\link[corpcor]{sda}}.
#' @param diagonal Logical, if \code{TRUE}, the method restricts the covariance matrix 
#'   to be diagonal. Defaults to \code{FALSE}.
#' @param verbose Logical, if \code{TRUE}, prints progress messages from \code{\link[corpcor]{sda}}. 
#'   Defaults to \code{FALSE}.
#'
#' @return A \code{\link[multivarious]{discriminant_projector}} object (subclass \code{"shrinkda"}) containing:
#' \itemize{
#'   \item \code{v} : A \code{d \times m} matrix of discriminant directions 
#'         (\code{ret\$beta} transposed).
#'   \item \code{s} : An \code{n \times m} matrix of projected scores (the original data 
#'         in the shrinkage discriminant space).
#'   \item \code{sdev} : Standard deviations of each column in \code{s}.
#'   \item \code{labels} : The class labels (in character form).
#'   \item \code{alpha} : Intercept or offset term from \code{\link[corpcor]{sda}} result (if any).
#'   \item \code{preproc} : The pre-processing object used.
#'   \item \code{classes} : Includes the string \code{"shrinkda"}.
#' }
#'
#' @details
#' Internally, the function:
#' \enumerate{
#'   \item Preprocesses \code{X} using \code{preproc}.
#'   \item Calls \code{\link[corpcor]{sda}} with the specified \code{lambda}, \code{lambda.var}, \code{lambda.freqs}, etc.
#'   \item Extracts \code{ret\$beta} (the discriminant directions) and \code{ret\$alpha} (the intercept).
#'   \item Projects the preprocessed data to obtain the score matrix \code{s}.
#'   \item Constructs a \code{discriminant_projector} object with class \code{"shrinkda"}.
#' }
#'
#' @seealso \code{\link[corpcor]{sda}}, \code{\link[multivarious]{discriminant_projector}}
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
  
  # 1) Preprocess the data
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  
  # 2) Call corpcor::sda
  ret <- corpcor::sda(Xp, Y, lambda, lambda.var, lambda.freqs, diagonal, verbose)
  
  # 3) Build score matrix = t(tcrossprod(ret$beta, Xp))
  #    Alternatively: crossprod(t(ret$beta), Xp), but you match the shape
  scores <- t(tcrossprod(ret$beta, Xp))  # (n x m)
  
  # 4) Return a discriminant_projector object
  multivarious::discriminant_projector(
    v       = t(ret$beta),            # (d x m) loadings
    s       = scores, 
    sdev    = apply(scores, 2, sd),
    preproc = procres,
    labels  = as.character(Y),
    alpha   = ret$alpha,              # offset (if used)
    classes = "shrinkda"
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
    chk::chk_equal(length(new_data), shape(x)[1])
    new_data <- matrix(new_data, byrow = TRUE, ncol = length(new_data))
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, values = nrow(coefficients(x)))
  
  # 1) Preprocess the new data
  nd_proc <- reprocess(x, new_data)
  
  # 2) Multiply by x$v
  out <- nd_proc %*% coefficients(x)
  
  # 3) Add intercept offset if stored
  if (!is.null(x$alpha)) {
    out <- out + x$alpha
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
  nd_proc <- reprocess(x, new_data, colind)
  
  # 3) Multiply by the corresponding rows in comp
  #    Then scale by nrow(comp)/length(colind)
  out <- nd_proc %*% comp[colind, ]
  
  out <- out * nrow(comp) / length(colind)
  
  # 4) Add alpha offset if present
  if (!is.null(x$alpha)) {
    out <- out + x$alpha
  }
  
  return(out)
}

