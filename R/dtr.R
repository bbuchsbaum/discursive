#' Discriminant analysis with Trace Regularization (DTR)
#' 
#' Finds a low-dimensional discriminant subspace that maximizes 
#' the between-class scatter while controlling the within-class scatter.
#'
#' @param X numeric matrix of predictors, of dimension n x p. 
#' @param Y factor variable of class labels, of length n.
#' @param preproc A preprocessing function to apply to the data. Default is centering.
#' @param d integer, the dimension of the discriminant subspace. Must be <= K-1 where K is the number of classes.
#' @param alpha numeric in \[0,1\] controlling the trade-off between between-class
#'   and within-class scatters.
#'
#' @return An S3 object of class "discriminant_projector" containing the transformation matrix W, 
#' the transformed scores, and related metadata.
#' 
#' @references 
#' Ahn, J., Chung, H. C., & Jeon, Y. (2021). Trace Ratio Optimization for High-Dimensional Multi-Class Discrimination. Journal of Computational and Graphical Statistics, 30(1), 192-203. \doi{10.1080/10618600.2020.1807352}
#' 
#' @examples
#' X = matrix(rnorm(100*1000), 100, 1000) 
#' y = sample(1:3, 100, replace=TRUE)
#' V = dtrda(X, y, d=2, alpha=0.5)
#' Xp = X %*% V  # project data onto discriminant subspace
#'
#' @export
dtrda <- function(X, Y,  preproc=multivarious::center(), d=2, alpha) {
  Y <- as.factor(Y)
  
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  
  n <- nrow(Xp)
  p <- ncol(Xp)
  K <- length(unique(Y))
  
  assertthat::assert_that(d <= K-1, "d must be less than the number of classes minus 1")
  assertthat::assert_that(alpha >= 0 && alpha <= 1,
                          msg = "alpha must be between 0 and 1")
  
  if (p > n) {
    # Compute orthonormal basis of the row space of Xc
    svd_res <- svd(Xp, nu = 0, nv = n - 1)
    P <- svd_res$v[, 1:(n - 1), drop = FALSE]
    # Project data onto row space
    Z <- Xp %*% P
  } else {
    Z <- Xp
  }
  
  # Compute scatter matrices in projected space
  M <- between_class_scatter(Z, Y)
  S <- within_class_scatter(Z, Y)
  
  # Form regularized matrix in projected space
  B <- (1-alpha)*M - alpha*S
  
  # Compute d leading eigenvectors of B
  eig <- eigen(B, symmetric = TRUE)
  U <- eig$vectors[,1:d,drop=FALSE]
  
  if (p > n) {
    # Back-transform eigenvectors to original space
    V <- P %*% U
  } else {
    V <- U
  }
  

  s <- Xp %*% V
  
  multivarious::discriminant_projector(v = V,
                                       s = s,
                                       sdev = apply(s, 2, sd),
                                       preproc = procres,
                                       labels = Y, 
                                       classes = "dtrda")
  
  
}