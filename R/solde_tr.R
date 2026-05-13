#' Stable and Orthogonal Local Discriminant Embedding via Trace Ratio
#'
#' Extension of the SOLDE-TR algorithm with `Rnanoflann`-based fast adjacency
#' construction. The method recasts the Stable Orthogonal Local Discriminant
#' Embedding (SOLDE) objective as a trace ratio optimisation and uses an
#' iterative trace-ratio (ITR) algorithm.
#'
#' @param X A numeric matrix `N x D` (rows = samples).
#' @param y A vector of length `N` of class labels.
#' @param ncomp Desired subspace dimension (default 10). Equivalent to `m` in
#'   earlier versions.
#' @param s Number of neighbours for adjacency construction; default 5.
#' @param alpha Numeric in `[0.5, 1]`. Balances `L_s` vs `L_v`; default 0.9.
#' @param sigma Positive scalar for the heat kernel; default 1.0.
#' @param tol Stopping threshold on the change in trace ratio; default `1e-6`.
#' @param maxit Maximum number of ITR iterations; default 100.
#' @param pca_preprocess Logical — perform PCA to ensure full rank; default
#'   `TRUE`.
#' @param preproc A `multivarious` preprocessing function. Defaults to
#'   [multivarious::center()].
#'
#' @return A `discriminant_projector` (subclass `"solde_tr"`) with an
#'   additional `trace_ratio_history` field.
#'
#' @references
#' Yang, X., Liu, G., Yu, Q., & Wang, R. (2017). "Stable and orthogonal local
#' discriminant embedding using trace ratio criterion for dimensionality
#' reduction." *Pattern Recognition*, 71, 249–264.
#'
#' @examples
#' \dontrun{
#' library(Rnanoflann)
#' set.seed(1)
#' X <- matrix(rnorm(100*20), 100, 20)
#' y <- sample.int(3, size = 100, replace = TRUE)
#' fit <- solde_tr(X, y, ncomp = 8, s = 5, alpha = 0.9)
#' }
#' @export
solde_tr <- function(X, y,
                     ncomp = 10,
                     s = 5,
                     alpha = 0.9,
                     sigma = 1.0,
                     tol = 1e-6,
                     maxit = 100,
                     pca_preprocess = TRUE,
                     preproc = multivarious::center()) {

  cl <- match.call()
  m <- ncomp

  stopifnot(requireNamespace("Rnanoflann", quietly = TRUE))
  if (!is.matrix(X)) X <- as.matrix(X)
  if (length(y) != nrow(X)) {
    stop("Length of label vector y must match the number of rows in X.")
  }
  procres <- multivarious::prep(preproc)
  X_proc <- multivarious::init_transform(procres, X)
  N <- nrow(X_proc)
  D <- ncol(X_proc)

  build_adjacency <- function(Xpca, y, s, sigma, mode = c("similarity", "diversity", "interclass")) {
    mode <- match.arg(mode)
    nobs <- nrow(Xpca)
    W <- matrix(0, nrow = nobs, ncol = nobs)

    suppressWarnings(
      nn_out <- Rnanoflann::nn(
        data    = Xpca,
        points  = Xpca,
        k       = s + 1,
        method  = "euclidean",
        search  = "standard",
        eps     = 0.0,
        square  = FALSE,
        trans   = FALSE,
        sorted  = FALSE
      )
    )

    idx_mat  <- nn_out$indices
    dist_mat <- nn_out$distances

    for (i in seq_len(nobs)) {
      neigh_idx  <- idx_mat[i, -1]
      neigh_dist <- dist_mat[i, -1]
      for (k_i in seq_along(neigh_idx)) {
        j  <- neigh_idx[k_i]
        dd <- neigh_dist[k_i]
        if (j < 1 || j > nobs) next

        same_class <- (y[i] == y[j])
        if (mode == "similarity") {
          if (same_class) {
            kij <- exp(-dd^2 / sigma)
            W[i, j] <- max(W[i, j], kij)
            W[j, i] <- max(W[j, i], kij)
          }
        } else if (mode == "diversity") {
          if (same_class) {
            kij <- exp(-dd^2 / sigma)
            val <- 1 - kij
            W[i, j] <- max(W[i, j], val)
            W[j, i] <- max(W[j, i], val)
          }
        } else if (mode == "interclass") {
          if (!same_class) {
            kij <- exp(-dd^2 / sigma)
            W[i, j] <- max(W[i, j], kij)
            W[j, i] <- max(W[j, i], kij)
          }
        }
      }
    }
    W
  }

  diag_colsum <- function(A) diag(colSums(A))

  X_centered <- X_proc
  W_pca <- diag(1, D, D)
  eigvals_pca <- rep(1, D)

  if (pca_preprocess) {
    Cov <- crossprod(X_centered) / (N - 1)
    eC <- eigen(Cov, symmetric = TRUE)
    vals <- eC$values
    vecs <- eC$vectors

    pos_idx <- which(vals > 1e-12)
    vals <- vals[pos_idx]
    vecs <- vecs[, pos_idx, drop = FALSE]
    o <- order(vals, decreasing = TRUE)
    vals <- vals[o]
    vecs <- vecs[, o, drop = FALSE]

    W_pca       <- vecs
    eigvals_pca <- vals
    X_pca <- X_centered %*% W_pca
  } else {
    X_pca <- X_centered
  }

  PC <- ncol(X_pca)
  if (PC < m) {
    stop("PCA subspace dimension < desired ncomp. Reduce ncomp or disable PCA.")
  }

  S_adj <- build_adjacency(X_pca, y, s, sigma, mode = "similarity")
  H     <- build_adjacency(X_pca, y, s, sigma, mode = "diversity")
  F_adj <- build_adjacency(X_pca, y, s, sigma, mode = "interclass")

  Ds <- diag_colsum(S_adj); L_s <- Ds - S_adj
  Dv <- diag_colsum(H);     L_v <- Dv - H
  Dm <- diag_colsum(F_adj); L_m <- Dm - F_adj

  L_d <- alpha * L_s - (1 - alpha) * L_v

  S_p  <- t(X_pca) %*% L_m %*% X_pca
  S_d_ <- t(X_pca) %*% L_d %*% X_pca
  S_p  <- 0.5 * (S_p + t(S_p))
  S_d_ <- 0.5 * (S_d_ + t(S_d_))

  diag(S_d_) <- diag(S_d_) + 1e-12

  W0 <- matrix(rnorm(PC * m), nrow = PC, ncol = m)
  Wk <- qr.Q(qr(W0))[, seq_len(m), drop = FALSE]

  trace_ratio_history <- numeric(maxit)

  for (iter in seq_len(maxit)) {
    tr_num <- sum(diag(t(Wk) %*% S_p  %*% Wk))
    tr_den <- sum(diag(t(Wk) %*% S_d_ %*% Wk))
    lambda_t <- tr_num / tr_den

    M <- S_p - lambda_t * S_d_
    eM <- eigen(M, symmetric = TRUE)
    idx_top <- order(eM$values, decreasing = TRUE)[seq_len(m)]
    Wnew <- eM$vectors[, idx_top, drop = FALSE]
    Wnew <- qr.Q(qr(Wnew))[, seq_len(m), drop = FALSE]

    trace_ratio_history[iter] <- lambda_t

    if (iter > 1 && abs(trace_ratio_history[iter] - trace_ratio_history[iter - 1]) < tol) {
      trace_ratio_history <- trace_ratio_history[seq_len(iter)]
      Wk <- Wnew
      break
    }
    Wk <- Wnew
  }

  W_solde_tr <- Wk
  W_final <- W_pca %*% W_solde_tr
  s_mat <- X_proc %*% W_final

  multivarious::discriminant_projector(
    v       = W_final,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = procres,
    labels  = y,
    trace_ratio_history = trace_ratio_history,
    method  = "solde_tr",
    call    = cl,
    classes = "solde_tr"
  )
}
