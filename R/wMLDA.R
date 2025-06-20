#' Weighted Multi-label Linear Discriminant Analysis (wMLDA)
#'
#' This function implements the Weighted Multi-label Linear Discriminant Analysis (wMLDA)
#' framework as described in the paper "A weighted linear discriminant analysis framework 
#' for multi-label feature extraction" by Jianhua Xu et al. The wMLDA framework unifies several
#' weight forms for multi-label LDA, including binary, correlation, entropy, fuzzy, and dependence-based
#' weighting. Each weighting strategy determines how much each instance contributes 
#' to each label, which in turn defines the between-class and within-class scatter matrices.
#'
#' The final result is returned as a \code{discriminant_projector} object from the \code{multivarious} package,
#' which can be integrated into downstream analytical workflows (e.g. applying the projection to new data).
#'
#' @param X A numeric matrix or data frame of size n x d, where n is the number of samples and d is the number of features.
#' @param Y A binary label matrix of size n x q, where Y[i, k] = 1 if sample i has label k, and 0 otherwise.
#' @param weight_method A character string specifying the weight form to use. One of:
#'   \itemize{
#'     \item "binary": Each relevant label gets weight 1, potentially over-counting for multi-label instances.
#'     \item "correlation": Uses global label correlation to determine weights, possibly assigning positive weights to irrelevant labels.
#'     \item "entropy": Weights are the reciprocal of the number of relevant labels, distributing weights evenly among relevant labels.
#'     \item "fuzzy": A fuzzy membership approach that uses both label and feature information.
#'     \item "dependence": A dependence-based form using Hilbert-Schmidt independence criterion (HSIC) and random block coordinate descent.
#'   }
#' @param ncomp The number of components (dimensions) to extract. Must be \eqn{\leq q-1}. Defaults to \code{q - 1}.
#' @param max_iter_fuzzy Maximum number of iterations for the fuzzy method. Default 100.
#' @param tol_fuzzy Convergence tolerance for the fuzzy method. Default 1e-6.
#' @param max_iter_dep Maximum number of epochs for the dependence-based RBCDM method. Default 100.
#' @param preproc A preprocessing step from \code{multivarious}, e.g. \code{center()} or \code{scale()}. Defaults to \code{center()}.
#' @param reg A small regularization value added to \code{Sw} to ensure invertibility. Default 1e-9.
#' @param seed Random seed for reproducibility. Default NULL (no setting of seed).
#'
#' @return A \code{discriminant_projector} object containing:
#'   \item{rotation}{The projection matrix (d x ncomp) mapping original features into discriminant space.}
#'   \item{s}{The projected scores of the training data (n x ncomp).}
#'   \item{sdev}{Standard deviations of the scores.}
#'   \item{labels}{The class (label) information.}
#'   \item{preproc}{The preprocessing object.}
#'   \item{classes}{The string "wMLDA".}
#'
#' @references 
#' Xu, J. "A weighted linear discriminant analysis framework for multi-label feature extraction." 
#' Knowledge-Based Systems, Volume 131, 2017, Pages 1-13.
#'
#' @examples
#' \dontrun{
#' library(multivarious)
#' set.seed(123)
#' X <- matrix(rnorm(100*5), nrow=100, ncol=5)
#' # Suppose we have 3 labels:
#' Y <- matrix(0, nrow=100, ncol=3)
#' # Assign random labels:
#' for (i in 1:100) {
#'   lab_count <- sample(1:3, 1)
#'   chosen <- sample(1:3, lab_count)
#'   Y[i, chosen] <- 1
#' }
#'
#' res <- wMLDA(X, Y, weight_method="entropy", ncomp=2)
#' str(res)
#' }
#' @export
wMLDA <- function(X, Y, weight_method=c("binary","correlation","entropy","fuzzy","dependence"),
                  ncomp=NULL, max_iter_fuzzy=100, tol_fuzzy=1e-6, max_iter_dep=100,
                  preproc = multivarious::center(), reg=1e-9, seed=NULL) {
  assertthat::assert_that(is.matrix(X), is.matrix(Y))
  weight_method <- match.arg(weight_method)
  
  n <- nrow(X)
  d <- ncol(X)
  q <- ncol(Y)
  if (is.null(ncomp)) {
    ncomp <- q - 1
  }
  assertthat::assert_that(ncomp <= q-1 && ncomp >= 1)
  
 
  if (!is.null(seed)) set.seed(seed)
  
  # Preprocessing using multivarious
  procres <- multivarious::prep(preproc)
  Xp <- multivarious::init_transform(procres, X)
  
  # Compute weight matrix
  Wmat <- switch(weight_method,
                 "binary" = weight_binary(Y),
                 "correlation" = weight_correlation(Y),
                 "entropy" = weight_entropy(Y),
                 "fuzzy" = weight_fuzzy(Xp, Y, max_iter=max_iter_fuzzy, tol=tol_fuzzy),
                 "dependence" = weight_dependence(Xp, Y, max_iter=max_iter_dep))
  
  w_sum_inst <- rowSums(Wmat) # \hat{w}_i
  
  # Compute scatter matrices
  w_sum_class <- colSums(Wmat)
  hat_n <- 1/w_sum_class
  hat_n[!is.finite(hat_n)] <- 0
  hat_W <- sweep(Wmat, 2, sqrt(hat_n), "*")
  
  D_w <- diag(w_sum_inst, n, n)
  
  M_sw <- D_w - hat_W %*% t(hat_W)
  M_sb <- (hat_W %*% matrix(w_sum_inst, nrow=1)) - (w_sum_inst %*% t(w_sum_inst))/n
  
  # Project into feature space
  Sw_M <- t(Xp) %*% M_sw %*% Xp
  Sb_M <- t(Xp) %*% M_sb %*% Xp
  
  # Solve generalized eigenvalue problem: Sb_M v = lambda Sw_M v
  # Regularize Sw_M
  Sw_M_reg <- Sw_M + diag(reg, d, d)
  
  M <- tryCatch(solve(Sw_M_reg, Sb_M),
                error=function(e) {
                  # fallback to pseudo-inverse if needed
                  pinvSw <- MASS::ginv(Sw_M_reg)
                  pinvSw %*% Sb_M
                })
  
  eig <- eigen(M, symmetric=TRUE)
  vals <- eig$values
  vecs <- eig$vectors
  
  idx <- order(vals, decreasing=TRUE)
  vals <- vals[idx]
  vecs <- vecs[, idx, drop=FALSE]
  
  # Select top ncomp
  W <- vecs[, 1:ncomp, drop=FALSE]
  
  # Scores
  s <- Xp %*% W
  
  # Return a discriminant_projector object
  multivarious::discriminant_projector(
    v = W,
    s = s,
    sdev = apply(s, 2, sd),
    preproc = procres,
    labels = Y,
    classes = "wMLDA"
  )
}


#############################
# Weighting Methods

#' @keywords internal
#' @noRd
weight_binary <- function(Y) {
  Y
}

#' @keywords internal
#' @noRd
weight_correlation <- function(Y) {
  l <- nrow(Y)
  q <- ncol(Y)

  norm_y <- sqrt(colSums(Y^2))
  norm_y[norm_y == 0] <- Inf

  C <- crossprod(Y) / tcrossprod(norm_y)
  C[!is.finite(C)] <- 0

  W <- t(C %*% t(Y))
  rsY <- rowSums(Y)
  non_zero <- rsY > 0
  W[non_zero, ] <- W[non_zero, , drop = FALSE] / rsY[non_zero]
  W[!non_zero, ] <- 0

  W[W < 0] <- 0
  W
}

#' @keywords internal
#' @noRd
weight_entropy <- function(Y) {
  rsY <- rowSums(Y)
  denom <- ifelse(rsY > 0, rsY, 1)
  W <- Y / denom
  W[rsY == 0, ] <- 0
  W
}

#' @keywords internal
#' @noRd
weight_fuzzy <- function(X, Y, max_iter=100, tol=1e-6) {
  # Initialize W using entropy-based weights
  W <- weight_entropy(Y)
  l <- nrow(Y)
  q <- ncol(Y)
  
  update_means <- function(X, W) {
    w2 <- W^2
    denom <- colSums(w2)
    M <- t(w2) %*% X
    non_zero <- denom > 0
    M[non_zero, ] <- M[non_zero, , drop = FALSE] / denom[non_zero]
    M[!non_zero, ] <- 0
    M
  }
  
  M <- update_means(X, W)
  oldW <- W
  for (iter in 1:max_iter) {
    # dist(i,k)^2
    dist2 <- matrix(0, l, q)
    for (k in 1:q) {
      diff <- sweep(X, 2, M[k,], "-")
      dist2[,k] <- rowSums(diff^2)
    }
    dist2[dist2 < 1e-12] <- 1e-12
    for (i in 1:l) {
      rel_labels <- which(Y[i,]==1)
      if (length(rel_labels)>0) {
        num <- Y[i,rel_labels]/dist2[i,rel_labels]
        denom <- sum(num)
        if (denom>0) {
          W[i,] <- 0
          W[i,rel_labels] <- num/denom
        } else {
          W[i,] <- 0
          W[i,rel_labels] <- 1/length(rel_labels)
        }
      } else {
        W[i,] <- 0
      }
    }
    M <- update_means(X, W)
    diff_val <- max(abs(W - oldW))
    if (diff_val < tol) break
    oldW <- W
  }
  
  W
}

#' @keywords internal
#' @noRd
weight_dependence <- function(X, Y, max_iter=100) {
  l <- nrow(Y)
  # HSIC-based RBCDM
  u <- rep(1,l)
  H <- diag(l) - (u %*% t(u))/l
  XXt <- X %*% t(X)
  Theta <- H %*% XXt %*% H
  
  # Initialize W with entropy:
  W <- weight_entropy(Y)
  
  for (epoch in 1:max_iter) {
    perm_idx <- sample(l)
    W_old <- W
    yw <- Y*W
    for (idx in perm_idx) {
      yi <- Y[idx,]
      rel_labels <- which(yi==1)
      if (length(rel_labels)==0) next
      # g_i(k) = sum_j Theta[i,j]*yw[j,k]
      gvals <- numeric(length(rel_labels))
      for (r in seq_along(rel_labels)) {
        k <- rel_labels[r]
        gvals[r] <- sum(Theta[idx,]*yw[,k])
      }
      k0 <- rel_labels[which.max(gvals)]
      w_new <- rep(0, ncol(Y))
      w_new[k0] <- 1
      W[idx,] <- w_new
    }
    
    if (all(W==W_old)) break
  }
  
  W
}