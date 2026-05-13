#' Weighted Multi-label Linear Discriminant Analysis (wMLDA)
#'
#' Implements the wMLDA framework (Xu, 2017). Unifies several weighting forms
#' for multi-label LDA (binary, correlation, entropy, fuzzy, dependence) that
#' define how each instance contributes to each label.
#'
#' @param X A numeric matrix `n x d`.
#' @param y A binary label matrix `n x q` where `y[i, k] = 1` if sample `i`
#'   has label `k`, and `0` otherwise.
#' @param ncomp Number of components to extract. Must be `<= q - 1`. Defaults
#'   to `q - 1`.
#' @param weight_method Character — one of `"binary"`, `"correlation"`,
#'   `"entropy"`, `"fuzzy"`, `"dependence"`.
#' @param max_iter_fuzzy Max iterations for `"fuzzy"` (default 100).
#' @param tol_fuzzy Convergence tolerance for `"fuzzy"` (default `1e-6`).
#' @param max_iter_dep Max epochs for `"dependence"` (default 100).
#' @param preproc Preprocessing from `multivarious`. Defaults to
#'   [multivarious::center()].
#' @param reg Small regularisation added to `Sw` for invertibility. Default
#'   `1e-9`.
#' @param seed Random seed for reproducibility (default `NULL`).
#'
#' @return A `discriminant_projector` (subclass `"wMLDA"`).
#'
#' @references
#' Xu, J. (2017). "A weighted linear discriminant analysis framework for
#' multi-label feature extraction." *Knowledge-Based Systems*, 131, 1–13.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' X <- matrix(rnorm(100*5), nrow = 100, ncol = 5)
#' y <- matrix(0, nrow = 100, ncol = 3)
#' for (i in seq_len(100)) {
#'   chosen <- sample(1:3, sample(1:3, 1))
#'   y[i, chosen] <- 1
#' }
#' res <- wMLDA(X, y, weight_method = "entropy", ncomp = 2)
#' print(res)
#' }
#' @export
wMLDA <- function(X, y,
                  ncomp = NULL,
                  weight_method = c("binary", "correlation", "entropy", "fuzzy", "dependence"),
                  max_iter_fuzzy = 100, tol_fuzzy = 1e-6, max_iter_dep = 100,
                  preproc = multivarious::center(), reg = 1e-9, seed = NULL) {

  cl <- match.call()
  assertthat::assert_that(is.matrix(X), is.matrix(y))
  weight_method <- match.arg(weight_method)

  n <- nrow(X)
  d <- ncol(X)
  q <- ncol(y)
  if (is.null(ncomp)) ncomp <- q - 1
  assertthat::assert_that(ncomp <= q - 1 && ncomp >= 1)

  if (!is.null(seed)) set.seed(seed)

  procres <- multivarious::prep(preproc)
  Xp <- multivarious::init_transform(procres, X)

  Wmat <- switch(weight_method,
                 "binary"      = weight_binary(y),
                 "correlation" = weight_correlation(y),
                 "entropy"     = weight_entropy(y),
                 "fuzzy"       = weight_fuzzy(Xp, y, max_iter = max_iter_fuzzy, tol = tol_fuzzy),
                 "dependence"  = weight_dependence(Xp, y, max_iter = max_iter_dep))

  w_sum_inst  <- rowSums(Wmat)
  w_sum_class <- colSums(Wmat)
  hat_n <- 1 / w_sum_class
  hat_n[!is.finite(hat_n)] <- 0
  hat_W <- sweep(Wmat, 2, sqrt(hat_n), "*")

  D_w <- diag(w_sum_inst, n, n)

  M_sw <- D_w - hat_W %*% t(hat_W)
  M_sb <- (hat_W %*% matrix(w_sum_inst, nrow = 1)) - (w_sum_inst %*% t(w_sum_inst)) / n

  Sw_M <- t(Xp) %*% M_sw %*% Xp
  Sb_M <- t(Xp) %*% M_sb %*% Xp

  Sw_M_reg <- Sw_M + diag(reg, d, d)

  M <- tryCatch(solve(Sw_M_reg, Sb_M),
                error = function(e) MASS::ginv(Sw_M_reg) %*% Sb_M)

  eig <- eigen(M, symmetric = TRUE)
  idx <- order(eig$values, decreasing = TRUE)
  vecs <- eig$vectors[, idx, drop = FALSE]

  W <- vecs[, seq_len(ncomp), drop = FALSE]
  s <- Xp %*% W

  multivarious::discriminant_projector(
    v       = W,
    s       = s,
    sdev    = apply(s, 2, sd),
    preproc = procres,
    labels  = y,
    method  = "wMLDA",
    call    = cl,
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
weight_fuzzy <- function(X, Y, max_iter = 100, tol = 1e-6) {
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
    dist2 <- matrix(0, l, q)
    for (k in 1:q) {
      diff <- sweep(X, 2, M[k, ], "-")
      dist2[, k] <- rowSums(diff^2)
    }
    dist2[dist2 < 1e-12] <- 1e-12
    for (i in 1:l) {
      rel_labels <- which(Y[i, ] == 1)
      if (length(rel_labels) > 0) {
        num <- Y[i, rel_labels] / dist2[i, rel_labels]
        denom <- sum(num)
        if (denom > 0) {
          W[i, ] <- 0
          W[i, rel_labels] <- num / denom
        } else {
          W[i, ] <- 0
          W[i, rel_labels] <- 1 / length(rel_labels)
        }
      } else {
        W[i, ] <- 0
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
weight_dependence <- function(X, Y, max_iter = 100) {
  l <- nrow(Y)
  u <- rep(1, l)
  H <- diag(l) - (u %*% t(u)) / l
  XXt <- X %*% t(X)
  Theta <- H %*% XXt %*% H

  W <- weight_entropy(Y)

  for (epoch in 1:max_iter) {
    perm_idx <- sample(l)
    W_old <- W
    yw <- Y * W
    for (idx in perm_idx) {
      yi <- Y[idx, ]
      rel_labels <- which(yi == 1)
      if (length(rel_labels) == 0) next
      gvals <- numeric(length(rel_labels))
      for (r in seq_along(rel_labels)) {
        k <- rel_labels[r]
        gvals[r] <- sum(Theta[idx, ] * yw[, k])
      }
      k0 <- rel_labels[which.max(gvals)]
      w_new <- rep(0, ncol(Y))
      w_new[k0] <- 1
      W[idx, ] <- w_new
    }
    if (all(W == W_old)) break
  }
  W
}
