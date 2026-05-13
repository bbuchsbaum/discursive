#' Compute Weighted Group Means
#'
#' Given a data matrix `X` (`n x d`) and a class-weight matrix `F`
#' (`c x n`), computes the weighted means of `X` for each class. Each
#' sample contributes to each class's mean according to its weight.
#'
#' @param X A numeric matrix of shape `n x d`. Coerced via `as.matrix()` if
#'   needed.
#' @param F A numeric matrix of shape `c x n`. `F[i, j]` is the weight of
#'   sample `j` for class `i`. Each row must have a positive sum.
#'
#' @return A numeric matrix of shape `c x d`.
#'
#' @examples
#' X <- matrix(1:10, nrow = 5, ncol = 2)
#' F <- matrix(runif(3*5), nrow = 3)
#' weighted_group_means(X, F)
#'
#' @export
weighted_group_means <- function(X, F) {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(F)) F <- as.matrix(F)

  row_sums_F <- rowSums(F)
  if (any(row_sums_F <= 0)) {
    stop("Each row of F must have a positive sum of weights (>=1 sample).")
  }

  ret <- F %*% X
  ret <- sweep(ret, 1, row_sums_F, "/")

  rownames(ret) <- rownames(F)
  colnames(ret) <- colnames(X)
  ret
}


#' Soft-Label Linear Discriminant Analysis (SL-LDA)
#'
#' A soft-label variant of LDA following Zhao et al. (2014). Instead of hard
#' (0/1) labels each sample carries fractional memberships across classes,
#' encoded in a matrix `y` (also called `C`).
#'
#' @param X A numeric matrix `n x d`.
#' @param y A numeric matrix `n x c` of soft class memberships (`y[i, k]` is
#'   the weight of sample `i` for class `k`). Must be non-negative. This was
#'   called `C` in earlier versions.
#' @param ncomp Integer. Final between-class projection dimension. Defaults
#'   to `ncol(y) - 1`. Equivalent to `dl` in earlier versions.
#' @param dp Integer. PCA dimension. Defaults to `min(dim(X))`.
#' @param di Integer. Within-class subspace dimension. Defaults to `dp - 1`.
#' @param alpha Numeric (`>= 0`) ridge added to the soft within-class scatter
#'   for invertibility. Default `0`.
#' @param preproc Preprocessing from `multivarious`. Defaults to
#'   [multivarious::pass()] (no centering).
#'
#' @return A `discriminant_projector` (subclass `"soft_lda"`).
#'
#' @references Zhao, M., Zhang, Z., Chow, T.W.S., & Li, B. (2014).
#'   "A general soft label based Linear Discriminant Analysis for
#'   semi-supervised dimensionality reduction." *Neurocomputing*, 135,
#'   250–264.
#'
#' @export
soft_lda <- function(X, y,
                     ncomp = ncol(y) - 1,
                     dp = min(dim(X)),
                     di = dp - 1,
                     alpha = 0.0,
                     preproc = multivarious::pass()) {

  cl <- match.call()
  C <- y
  dl <- ncomp

  if (!is.matrix(X)) stop("X must be a matrix.")
  if (!is.matrix(C)) stop("y must be a matrix of soft membership weights.")
  if (nrow(C) != nrow(X)) {
    stop("y must have the same number of rows as X.")
  }
  if (any(C < 0)) {
    stop("All entries of 'y' must be non-negative (soft memberships).")
  }

  n <- nrow(X)
  d <- ncol(X)
  c <- ncol(C)

  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)

  pca_red <- multivarious::pca(Xp, ncomp = dp)
  proj_dp <- pca_red$v
  Xpca    <- scores(pca_red)

  F <- t(C)
  E <- Matrix::Diagonal(x = rowSums(C))
  G <- diag(colSums(C))

  e_vec <- rep(1, n)
  denom <- as.numeric(t(e_vec) %*% diag(E) %*% e_vec)

  Ft   <- t(F)
  invG <- diag(1 / diag(G))
  M_sw <- E - Ft %*% invG %*% F
  M_sb <- Ft %*% invG %*% F - (E %*% tcrossprod(e_vec) %*% E) / denom

  Sw_raw <- crossprod(Xpca, M_sw %*% Xpca)
  Sw     <- Sw_raw + alpha * diag(dp)

  if (di > dp) {
    warning("di > dp is invalid; setting di = dp.")
    di <- dp
  }
  if (di < 1) stop("di must be >= 1.")

  E_i <- RSpectra::eigs_sym(Sw, k = di, which = "LM")
  proj_di <- E_i$vectors %*% diag(1 / sqrt(E_i$values))

  group_means_pca <- weighted_group_means(Xpca, F)
  gm_proj <- group_means_pca %*% proj_di

  if (dl > di) {
    warning("ncomp > di is invalid; setting ncomp = di.")
    dl <- di
  }
  if (dl < 1) stop("ncomp must be >= 1.")

  E_l <- multivarious::pca(gm_proj, ncomp = dl)
  proj_dl <- E_l$v

  proj_final <- proj_dp %*% proj_di %*% proj_dl
  s_mat <- Xpca %*% proj_di %*% proj_dl

  multivarious::discriminant_projector(
    v       = proj_final,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = procres,
    labels  = colnames(C),
    method  = "soft_lda",
    call    = cl,
    classes = "soft_lda"
  )
}
