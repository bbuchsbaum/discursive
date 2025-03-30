#' @title Apply PCA for Null Space Removal
#' @description Preprocess the data matrix to remove the null space using PCA 
#' and retain a number of principal components that explain at least 
#' `var_retained` proportion of variance.
#' @param X Data matrix of size n x d (n samples, d features).
#' @param var_retained Proportion of variance to retain (default is 0.95).
#' @return A list containing:
#' \item{X_pca}{The transformed data matrix after PCA (n x d_pca).}
#' \item{rot}{The PCA rotation/loadings matrix (d x d_pca).}
#' \item{fit}{The PCA fit object from `multivarious::pca`.}
apply_pca <- function(X, var_retained = 0.95) {
  pca_result <- multivarious::pca(X, preproc = multivarious::pass())
  # pca_result$d are singular values
  var_explained <- pca_result$d^2 / sum(pca_result$d^2)
  cum_var <- cumsum(var_explained)
  ncomp <- which(cum_var >= var_retained)[1]
  
  # In case var_retained is very small, ensure at least 1 component
  if (is.na(ncomp)) {
    ncomp <- length(var_explained)
  }
  
  X_pca <- pca_result$s[, 1:ncomp, drop = FALSE]
  list(X_pca = X_pca, rot = pca_result$v[, 1:ncomp, drop = FALSE], fit = pca_result)
}

#' @title Initialize Parameters for ALLDA
#' @description Initialize the similarity matrix S based on k-nearest neighbors.
#' @param X Data matrix of size n x d (n samples, d features).
#' @param y Label vector of length n (not directly used here but kept for consistency).
#' @param ncomp Reduced dimension (not used here but included for consistency).
#' @param k Number of neighbors.
#' @return A list containing the initialized similarity matrix S.
initialize_parameters <- function(X, y, ncomp, k) {
  n <- nrow(X)  # Number of samples
  S <- matrix(0, n, n)
  for (i in 1:n) {
    distances <- rowSums((X[i, , drop=FALSE] - X)^2)
    nn_idx <- order(distances)[2:(k + 1)]  # Exclude the point itself
    S[i, nn_idx] <- 1 / k
  }
  list(S = S)
}

#' @title Update Laplacian Matrix
#' @description Compute the Laplacian matrix L_A from the similarity matrix S.
#' @param S Similarity matrix (n x n).
#' @return Laplacian matrix L_A (n x n).
update_laplacian <- function(S) {
  D_A <- diag(rowSums(S))
  L_A <- D_A - (S + t(S)) / 2
  L_A
}

#' @title Update Transformation Matrix W
#' @description Solve for the transformation matrix W using eigen-decomposition.
#' @param X Data matrix of size n x d (n samples, d features).
#' @param L_A Laplacian matrix of size n x n.
#' @param ncomp Reduced dimension.
#' @param reg Regularization term to ensure invertibility of St (default is 1e-5).
#' @return Transformation matrix W (d x ncomp).
update_W <- function(X, L_A, ncomp, reg = 1e-5) {
  St <- t(X) %*% X  # d x d
  St <- St + reg * Matrix::Diagonal(ncol(St))  # Regularization
  
  Xt_LA_X <- t(X) %*% L_A %*% X  # d x d
  
  M <- solve(St) %*% Xt_LA_X
  eig <- RSpectra::eigs(M, k = ncomp, which = "SM")
  ord <- order(eig$values)
  
  W <- eig$vectors[, ord[1:ncomp], drop=FALSE]
  
  normfac <- sqrt(Matrix::diag(t(W) %*% St %*% W))
  W <- sweep(W, 2, normfac, "/")
  return(W)
}

#' @title Update Similarity Matrix S
#' @description Update the similarity matrix S using the optimal subspace projection W.
#' @param X Data matrix of size n x d (n samples, d features).
#' @param W Transformation matrix (d x ncomp).
#' @param k Number of neighbors.
#' @param r Parameter r controlling the influence of distances (should be > 1).
#' @param epsilon Small constant to avoid division by zero (default is 1e-10).
#' @return Updated similarity matrix S (n x n).
#' @importFrom Rnanoflann nn
update_similarity <- function(X, W, k, r, epsilon = 1e-10) {
  n <- nrow(X)
  S <- matrix(0, n, n)
  one_over_1_minus_r <- 1 / (1 - r)
  
  X_transformed <- X %*% W
  nn_result <- nn(data = X_transformed, points = X_transformed, k = k + 1, 
                  method = "euclidean", search = "standard", eps = 0.0, square = FALSE, trans = TRUE)
  
  indices <- nn_result$indices[, -1, drop=FALSE]
  distances <- nn_result$distances[, -1, drop=FALSE] + epsilon
  
  weights <- distances^one_over_1_minus_r
  weights_sum <- rowSums(weights)
  weights_normalized <- sweep(weights, 1, weights_sum, FUN = "/")
  
  for (i in 1:n) {
    S[i, indices[i, ]] <- weights_normalized[i, ]
  }
  
  return(S)
}

#' @title Adaptive Local Linear Discriminant Analysis (ALLDA)
#' @description Perform dimensionality reduction using the ALLDA algorithm.
#' @param X Data matrix of size n x d (n samples, d features).
#' @param y Label vector of length n.
#' @param ncomp Reduced dimension (must be less than the number of features retained by PCA).
#' @param k Number of neighbors.
#' @param r Parameter r (default is 2, must be > 1).
#' @param preproc A preprocessing step from `multivarious`. Defaults to centering.
#' @param max_iter Maximum number of iterations (default is 30).
#' @param tol Convergence tolerance (default is 1e-4).
#' @param var_retained Proportion of variance to retain during PCA (default is 0.95).
#' @param reg Regularization term to ensure invertibility of St (default is 1e-5).
#' @return An S3 object of class "discriminant_projector" containing the transformation matrix W, 
#' the transformed scores, and related metadata.
#' @references Nie, F., Wang, Z., Wang, R., Wang, Z., & Li, X. (2020). Adaptive local linear discriminant analysis. 
#' ACM Transactions on Knowledge Discovery from Data (TKDD), 14(1), 1-19.
#' @examples
#' # result <- allda(X, y, ncomp = 2, k = 5)
#' # W <- result$rotation
#' # S <- result$S
allda <- function(X, y, ncomp, k, r = 2, preproc=multivarious::center(),
                  max_iter = 30, tol = 1e-4, var_retained = 0.95, reg = 1e-5) {
  
  # Basic assertions
  assertthat::assert_that(ncomp > 0, msg="ncomp must be greater than 0")
  assertthat::assert_that(k > 0 && k < nrow(X), msg="k must be between 1 and n-1")
  assertthat::assert_that(r > 1, msg="r must be greater than 1")
  
  # Preprocessing
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  
  # Apply PCA to remove the null space
  pca_result <- apply_pca(Xp, var_retained)
  X_pca <- pca_result$X_pca
  
  assertthat::assert_that(ncomp < ncol(X_pca),
                          msg="ncomp must be less than the number of retained PCA components")
  
  # Initialize parameters
  init_params <- initialize_parameters(X_pca, y, ncomp, k)
  S <- init_params$S
  
  # Iterative updates
  for (iter in 1:max_iter) {
    L_A <- update_laplacian(S)
    W <- update_W(X_pca, L_A, ncomp, reg)
    S_new <- update_similarity(X_pca, W, k, r)
    
    if (norm(S_new - S, "F") < tol) {
      S <- S_new
      break
    }
    S <- S_new
  }
  
  # Map W back to original space
  W_final <- pca_result$rot %*% W
  scores <- X %*% W_final
  # Use multivarious to create a discriminant_projector
  multivarious::discriminant_projector(W_final, scores, apply(scores,2,sd),
                                       preproc=procres,
                                       labels=as.character(y),
                                       adjacency=S,
                                       classes="allda")
}

