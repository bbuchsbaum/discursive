#' PCA followed by Linear Discriminant Analysis
#'
#' This function applies Principal Component Analysis (PCA) followed by Linear Discriminant Analysis (LDA) to a given dataset.
#' The data is first projected onto \code{dp} principal components, then further transformed via a two-step LDA procedure:
#' an intermediate within-class projection of dimension \code{di}, followed by a final between-class projection of dimension \code{dl}.
#' This sequence of transformations aims to reduce dimensionality while enhancing class separability.
#'
#' @param X A numeric matrix of size \code{n x d}, where \code{n} is the number of samples (rows)
#'          and \code{d} is the number of features (columns).
#' @param Y A factor or numeric vector of length \code{n} representing class labels for each sample.
#'          If numeric, it will be converted to a factor.
#' @param preproc A preprocessing function from the \code{multivarious} package (e.g. \code{center()}, \code{scale()})
#'                to apply to the data before PCA. Defaults to centering.
#' @param dp Integer. The dimension of the initial PCA projection. Defaults to \code{min(dim(X))}, i.e.,
#'           the smaller of the number of samples or features. Must be at least 2 and at most \code{min(n,d)}.
#' @param di Integer. The dimension of the within-class projection, typically \code{dp-1}. Defaults to \code{dp-1}.
#' @param dl Integer. The dimension of the between-class projection. Defaults to \code{length(unique(Y))-1}, which
#'           is often the maximum number of discriminative axes for LDA.
#' @return An object of class \code{discriminant_projector} (from \code{multivarious}) containing:
#' \itemize{
#'   \item \code{rotation}: The final projection matrix of size \code{d x dl}, mapping from original features to \code{dl}-dimensional space.
#'   \item \code{s}: The projected data scores of size \code{n x dl}, where each row is a sample in the reduced space.
#'   \item \code{sdev}: The standard deviations of each dimension in the projected space.
#'   \item \code{labels}: The class labels associated with each sample.
#'   \item \code{dp}, \code{di}, \code{dl}: The specified or inferred PCA/LDA dimensions.
#'   \item \code{preproc}: The preprocessing object used.
#' }
#' @details
#' The function proceeds through the following steps:
#' \enumerate{
#'   \item \strong{Preprocessing}: The data \code{X} is preprocessed using the specified \code{preproc} function.
#'   \item \strong{PCA Projection}: The preprocessed data is projected onto the first \code{dp} principal components.
#'   \item \strong{Within-Class Scatter}: The within-class scatter matrix \code{Sw} is computed in the PCA-transformed space.
#'   \item \strong{Within-Class Projection}: The eigen-decomposition of \code{Sw} is used to derive an intermediate projection of dimension \code{di}.
#'   \item \strong{Between-Class Projection}: The projected group means are subjected to PCA to derive a final projection of dimension \code{dl}.
#'   \item \strong{Final Projection}: The data is ultimately projected onto the \code{dl}-dimensional subspace that maximizes class separation.
#' }
#' 
#' @seealso \code{\link[multivarious]{pca}}, \code{\link[RSpectra]{eigs_sym}}
#' 
#' @export
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' Y <- iris[, 5]
#' # Reduce to a space of dp=4, di=3, dl=2 for illustration
#' res <- pca_lda(X, Y, di=3)
#' }
pca_lda <- function(X, Y, preproc = center(), dp = min(dim(X)), di = dp - 1, dl = length(unique(Y)) - 1) {
  # Basic checks on dimensions
  chk::chk_range(ncol(X), c(2, 10e6))
  chk::chk_range(nrow(X), c(2, 10e6))
  chk::chk_range(dp, c(2, min(dim(X))))
  chk::chk_range(di, c(2, dp - 1))
  chk::chk_range(dl, c(1, length(unique(Y)) - 1))
  
  # Ensure Y is a factor
  Y <- as.factor(Y)
  
  # Preprocess data
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  
  # PCA to dp components
  pca_basis <- multivarious::pca(Xp, ncomp = dp)
  proj_dp <- pca_basis$v    # d x dp loadings
  Xpca <- scores(pca_basis) # n x dp scores
  
  # Compute within-class scatter matrix in PCA space
  Sw <- within_class_scatter(Xpca, Y) # dp x dp
  
  # Compute group means in PCA space
  gmeans <- group_means(Y, Xpca) # G x dp (G = number of groups)
  
  # Within-class projection (di dimensions)
  E_i <- RSpectra::eigs_sym(Sw, k = di)
  # Construct the within-class projection (dp x di)
  # Add a small tolerance to avoid division by zero when eigenvalues are
  # numerically close to zero
  proj_di <- E_i$vectors %*% diag(1 / sqrt(pmax(E_i$values, .Machine$double.eps)))
  
  # Project group means using the within-class projection
  gmeans_proj <- gmeans %*% proj_di # G x di
  
  # Between-class projection (dl dimensions) via PCA on group means projection
  E_l <- multivarious::pca(gmeans_proj, ncomp = dl)
  proj_dl <- E_l$v # di x dl
  
  # Final projection (d x dl)
  proj_final <- proj_dp %*% proj_di %*% proj_dl
  
  # Project original preprocessed data
  s <- Xp %*% proj_final # n x dl
  
  # Return a discriminant_projector object
  ret <- multivarious::discriminant_projector(
    v = proj_final,
    s = s,
    sdev = apply(s, 2, sd),
    dp = dp,
    dl = dl,
    di = di,
    labels = Y,
    classes = "pca_lda",
    preproc = procres
  )
  
  return(ret)
}




