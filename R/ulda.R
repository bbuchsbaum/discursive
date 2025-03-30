#' Uncorrelated Linear Discriminant Analysis (ULDA)
#'
#' This function performs Uncorrelated Linear Discriminant Analysis (ULDA) on a given dataset.
#' ULDA seeks linear combinations of features that provide maximum class separation while ensuring
#' that the resulting discriminant axes are mutually uncorrelated. It is particularly useful for
#' dimensionality reduction and classification tasks in high-dimensional spaces.
#'
#' @param X A numeric matrix of size \code{n x d}, where \code{n} is the number of samples (rows)
#'          and \code{d} is the number of features (columns).
#' @param Y A factor or numeric vector of length \code{n} representing class labels for each sample.
#'          If numeric, it will be converted to a factor.
#' @param preproc A preprocessing function from the \code{multivarious} package, such as \code{center()} or \code{scale()},
#'        to apply to the data before ULDA. The default is \code{center()}, which centers the columns of \code{X}.
#' @param mu A regularization parameter (currently unused, but included for extensibility). Default is 0.
#' @param tol A numeric tolerance level. Singular values smaller than \code{tol} are considered negligible and discarded.
#'        Default is \code{1e-6}.
#' @return An object of class \code{discriminant_projector} from \code{multivarious}, which contains:
#' \itemize{
#'   \item \code{rotation}: The final projection matrix mapping the original feature space to the ULDA space.
#'   \item \code{s}: The scores (projected data) of size \code{n x r}, where \code{r} is the reduced dimension.
#'   \item \code{sdev}: The standard deviations of the projected components.
#'   \item \code{labels}: The class labels.
#'   \item \code{preproc}: The preprocessing object used.
#'   \item \code{classes}: A string "ulda" indicating the type of projector.
#' }
#'
#' @details
#' The procedure for ULDA can be summarized as follows:
#' \enumerate{
#'   \item \strong{Preprocessing}: The data \code{X} is preprocessed using the specified \code{preproc} function.
#'   \item \strong{Class Statistics}: Compute class probabilities, class means, and the global mean of the preprocessed data.
#'   \item \strong{Between-Class Scatter}: Form a matrix \code{Hb} that captures between-class differences weighted by class probabilities.
#'   \item \strong{Total Scatter}: Form the total scatter matrix via the centered data \code{Ht}.
#'   \item \strong{SVD of Total Scatter}: Perform Singular Value Decomposition (SVD) on \code{Ht} to capture the most significant directions of variability.
#'   \item \strong{Projection onto Between-Class Structure}: Use SVD on a transformed version of \code{Hb} to find directions that maximize class separation.
#'   \item \strong{Final Projection}: Combine the transformations to yield a projection matrix whose columns are discriminant vectors that are uncorrelated.
#'   \item \strong{Projection of Data}: Project the preprocessed data onto these discriminant vectors to get the final scores.
#' }
#'
#' This approach ensures that the resulting discriminants are uncorrelated linear combinations of features.
#'
#' @seealso \code{\link[multivarious]{prep}}, \code{\link[RSpectra]{eigs_sym}}, \code{\link[multivarious]{discriminant_projector}}
#'
#' @importFrom multivarious prep
#' @export
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' Y <- iris[, 5]
#' res <- ulda(X, Y)
#' }
ulda <- function(X, Y, preproc = center(), mu = 0, tol = 1e-6) {
  # Ensure Y is a factor
  Y <- as.factor(Y)
  
  # Preprocess data
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  
  levs <- levels(Y)
  nc <- length(levs)
  
  # Class probabilities and means
  cprobs <- table(Y) / length(Y)
  cmeans <- group_means(Y, Xp)
  gmean <- colMeans(Xp)
  
  # Construct Hb capturing between-class variation
  Hb <- do.call(cbind, lapply(1:nc, function(i) {
    sqrt(cprobs[i]) * (cmeans[i, ] - gmean)
  }))
  
  # Center data around global mean for total scatter matrix
  Ht <- t(sweep(Xp, 2, gmean, "-"))
  
  # SVD of total scatter
  svd_ht <- svd(Ht)
  keep <- which(svd_ht$d > tol)
  
  # Transform Hb using the pseudoinverse of Ht
  B <- diag(1 / svd_ht$d[keep]) %*% t(svd_ht$u[, keep]) %*% Hb
  
  # SVD of B to find discriminant directions
  svd_B <- svd(B)
  keep_b <- which(svd_B$d > tol)
  
  # Compute final projection vectors
  vecs <- svd_ht$u[, keep, drop = FALSE] %*% diag(1 / svd_ht$d[keep]) %*% svd_B$u[, keep_b, drop = FALSE]
  
  # Project the data
  s <- Xp %*% vecs
  
  # Return a discriminant_projector object
  multivarious::discriminant_projector(
    v = vecs,
    s = s,
    sdev = apply(s, 2, sd),
    preproc = procres,
    labels = Y,
    classes = "ulda"
  )
}
