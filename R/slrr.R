#' Sparse or Ridge Low-Rank Regression (SLRR / LRRR)
#'
#' Implements the low-rank regression approach from Cai, Ding & Huang (2013).
#' Unifies:
#' \itemize{
#'   \item **Low-Rank Ridge Regression (LRRR)** (`penalty = "ridge"`) — adds a
#'   Frobenius-norm penalty.
#'   \item **Sparse Low-Rank Regression (SLRR)** (`penalty = "l21"`) — uses an
#'   `l_{2,1}` row-sparsity penalty via iterative reweighting.
#' }
#'
#' @param X A numeric matrix `n x d`.
#' @param y A factor or numeric vector of length `n`. If numeric, it is
#'   coerced to a factor.
#' @param ncomp Integer. Rank (subspace dimension) of the low-rank coefficient
#'   matrix `W`. Must be at most `K - 1` (where `K` is the number of classes)
#'   and at most `d`.
#' @param lambda Numeric penalty parameter. Default `0.001`.
#' @param penalty Either `"ridge"` (LRRR) or `"l21"` (SLRR).
#' @param max_iter Maximum number of iterations for `"l21"`. Ignored for
#'   `"ridge"`.
#' @param tol Convergence tolerance for `"l21"`.
#' @param preproc Preprocessing from `multivarious`. Defaults to
#'   [multivarious::center()].
#' @param verbose Logical. Print iteration details for `"l21"`.
#' @param st_ridge Small ridge added to `S_t` to avoid singularity (default
#'   `1e-6`).
#'
#' @return A `discriminant_projector` (subclass `"slrr"`) with additional
#'   fields `A` (subspace) and `B` (in-subspace regression).
#'
#' @references
#' Cai, X., Ding, C., & Huang, H. (2013). "On The Equivalent of Low-Rank
#' Regressions and Linear Discriminant Analysis Based Regressions." *KDD'13*.
#'
#' @export
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris[, 5]
#' fit_lrrr <- slrr(X, y, ncomp = 2, lambda = 0.01, penalty = "ridge")
#' fit_slrr <- slrr(X, y, ncomp = 2, lambda = 0.01, penalty = "l21")
#' }
slrr <- function(X, y, ncomp,
                 lambda = 0.001,
                 penalty = c("ridge", "l21"),
                 max_iter = 50,
                 tol = 1e-6,
                 preproc = multivarious::center(),
                 verbose = FALSE,
                 st_ridge = 1e-6) {

  cl <- match.call()
  penalty <- match.arg(penalty)
  s <- ncomp

  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  n <- nrow(Xp)
  d <- ncol(Xp)

  y <- as.factor(y)
  G <- model.matrix(~ y - 1)
  G <- sweep(G, 2, colSums(G), FUN = "/")
  cdim <- ncol(G)

  if (cdim < 2) {
    stop("y must contain at least two classes")
  }

  max_s <- min(cdim - 1, d)
  if (s < 1 || s > max_s) {
    stop(sprintf("'ncomp' must be between 1 and %d", max_s))
  }

  Xt  <- t(Xp)
  XtX <- Xt %*% Xp

  mu <- colMeans(Xp)
  St <- total_scatter(Xp, mu)
  Sb <- between_class_scatter(Xp, y)

  St_reg <- St + st_ridge * diag(d)
  M <- tryCatch(solve(St_reg, Sb),
                error = function(e) MASS::ginv(St_reg) %*% Sb)
  eigres <- eigen(M, symmetric = TRUE)
  A <- eigres$vectors[, seq_len(s), drop = FALSE]

  get_ABW <- function(A_in, D_in) {
    reg_mat  <- XtX + lambda * D_in
    left_mat <- t(A_in) %*% reg_mat %*% A_in
    left_inv <- tryCatch(solve(left_mat),
                         error = function(e) MASS::ginv(left_mat))
    B_in <- left_inv %*% (t(A_in) %*% Xt %*% G)
    W_in <- A_in %*% B_in
    list(A = A_in, B = B_in, W = W_in)
  }

  Dmat <- diag(d)

  if (penalty == "ridge") {
    outABW <- get_ABW(A, Dmat)
    W <- outABW$W
    B <- outABW$B
  } else {
    old_obj <- Inf
    W <- matrix(0, d, cdim)

    for (iter in seq_len(max_iter)) {
      St_iter <- St + lambda * Dmat + st_ridge * diag(d)
      M_l21 <- tryCatch(solve(St_iter, Sb),
                        error = function(e) MASS::ginv(St_iter) %*% Sb)
      eig_l21 <- eigen(M_l21, symmetric = TRUE)
      A_new <- eig_l21$vectors[, seq_len(s), drop = FALSE]

      reg_mat  <- XtX + lambda * Dmat
      left_mat <- t(A_new) %*% reg_mat %*% A_new
      left_inv <- tryCatch(solve(left_mat),
                           error = function(e) MASS::ginv(left_mat))
      B_new <- left_inv %*% (t(A_new) %*% Xt %*% G)
      W_new <- A_new %*% B_new

      row_norms <- sqrt(rowSums(W_new^2)) + 1e-12
      D_diag    <- 1 / (2 * row_norms)
      Dmat_new  <- diag(D_diag, d)

      s_mat_l21 <- Xp %*% W_new
      R    <- G - s_mat_l21
      fval1 <- sum(R^2)
      fval2 <- lambda * sum(row_norms)
      new_obj <- fval1 + fval2

      if (verbose) {
        message(sprintf("Iter=%d, obj=%.6f (fval=%.6f + pen=%.6f)",
                        iter, new_obj, fval1, fval2))
      }

      if (abs(old_obj - new_obj) < tol * (1 + abs(old_obj))) {
        A    <- A_new
        W    <- W_new
        B    <- B_new
        Dmat <- Dmat_new
        break
      }

      A    <- A_new
      W    <- W_new
      B    <- B_new
      Dmat <- Dmat_new
      old_obj <- new_obj

      if (iter == max_iter && verbose) {
        message("Max iterations reached in slrr l21 approach.")
      }
    }
  }

  s_mat <- Xp %*% W

  multivarious::discriminant_projector(
    v       = W,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = procres,
    labels  = y,
    A       = A,
    B       = B,
    method  = "slrr",
    call    = cl,
    classes = c("slrr", penalty)
  )
}
