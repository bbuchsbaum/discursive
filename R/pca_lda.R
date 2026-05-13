#' PCA followed by Linear Discriminant Analysis
#'
#' Applies PCA followed by LDA: project the data onto `dp` principal
#' components, then perform a two-step LDA — an intermediate within-class
#' projection of dimension `di` followed by a final between-class projection
#' of dimension `ncomp` (a.k.a. `dl`).
#'
#' @param X A numeric matrix of size `n x d`.
#' @param y A factor or numeric vector of length `n` of class labels. If
#'   numeric, it is coerced to a factor.
#' @param ncomp Integer. Final between-class projection dimension. Defaults to
#'   `length(unique(y)) - 1`. Equivalent to the `dl` argument in earlier
#'   versions; you may pass either name.
#' @param dp Integer. PCA projection dimension. Defaults to `min(dim(X))`.
#'   Must be in `[2, min(n, d)]`.
#' @param di Integer. Within-class projection dimension, typically `dp - 1`.
#'   Defaults to `dp - 1`.
#' @param preproc A preprocessing function from `multivarious`. Defaults to
#'   [multivarious::center()].
#'
#' @return A `discriminant_projector` (subclass `"pca_lda"`) with the
#'   additional fields `dp`, `di`, `dl` (with `dl = ncomp`).
#'
#' @seealso [multivarious::pca()], [RSpectra::eigs_sym()]
#'
#' @export
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris[, 5]
#' res <- pca_lda(X, y, di = 3)
#' }
pca_lda <- function(X, y, ncomp = length(unique(y)) - 1,
                    dp = min(dim(X)),
                    di = dp - 1,
                    preproc = multivarious::center()) {

  cl <- match.call()
  dl <- ncomp

  chk::chk_range(ncol(X), c(2, 10e6))
  chk::chk_range(nrow(X), c(2, 10e6))
  chk::chk_range(dp, c(2, min(dim(X))))
  chk::chk_range(di, c(2, dp - 1))
  chk::chk_range(dl, c(1, length(unique(y)) - 1))

  y <- as.factor(y)

  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)

  pca_basis <- multivarious::pca(Xp, ncomp = dp)
  proj_dp   <- pca_basis$v
  Xpca      <- scores(pca_basis)

  Sw <- within_class_scatter(Xpca, y)
  gmeans <- group_means(y, Xpca)

  E_i <- RSpectra::eigs_sym(Sw, k = di)
  proj_di <- E_i$vectors %*% diag(1 / sqrt(pmax(E_i$values, .Machine$double.eps)))

  gmeans_proj <- gmeans %*% proj_di

  E_l <- multivarious::pca(gmeans_proj, ncomp = dl)
  proj_dl <- E_l$v

  proj_final <- proj_dp %*% proj_di %*% proj_dl
  s <- Xp %*% proj_final

  multivarious::discriminant_projector(
    v       = proj_final,
    s       = s,
    sdev    = apply(s, 2, sd),
    dp      = dp,
    dl      = dl,
    di      = di,
    labels  = y,
    preproc = procres,
    method  = "pca_lda",
    call    = cl,
    classes = "pca_lda"
  )
}
