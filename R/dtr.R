#' Discriminant Analysis with Trace Regularisation (DTR)
#'
#' Finds a low-dimensional discriminant subspace that maximises between-class
#' scatter while controlling within-class scatter.
#'
#' @param X A numeric matrix of size `n x p` (rows = samples, columns = features).
#' @param y A factor or numeric vector of class labels of length `n`.
#' @param ncomp Integer. Dimension of the discriminant subspace. Must be
#'   `<= K - 1` where `K` is the number of classes. Default 2.
#' @param alpha Numeric in `[0, 1]` controlling the trade-off between
#'   between-class and within-class scatters.
#' @param preproc A preprocessing function from `multivarious`. Defaults to
#'   [multivarious::center()].
#'
#' @return A `discriminant_projector` object (subclass `"dtrda"`).
#'
#' @references
#' Ahn, J., Chung, H. C., & Jeon, Y. (2021). Trace Ratio Optimization for
#' High-Dimensional Multi-Class Discrimination. *Journal of Computational and
#' Graphical Statistics*, 30(1), 192–203.
#' \doi{10.1080/10618600.2020.1807352}
#'
#' @examples
#' X <- matrix(rnorm(100*1000), 100, 1000)
#' y <- sample(1:3, 100, replace = TRUE)
#' fit <- dtrda(X, y, ncomp = 2, alpha = 0.5)
#'
#' @export
dtrda <- function(X, y, ncomp = 2, alpha,
                  preproc = multivarious::center()) {
  cl <- match.call()
  y <- as.factor(y)

  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)

  n <- nrow(Xp)
  p <- ncol(Xp)
  K <- length(unique(y))

  assertthat::assert_that(ncomp <= K - 1,
                          msg = "'ncomp' must be <= number of classes - 1")
  assertthat::assert_that(alpha >= 0 && alpha <= 1,
                          msg = "'alpha' must be between 0 and 1")

  if (p > n) {
    svd_res <- svd(Xp, nu = 0, nv = n - 1)
    P <- svd_res$v[, 1:(n - 1), drop = FALSE]
    Z <- Xp %*% P
  } else {
    Z <- Xp
  }

  M <- between_class_scatter(Z, y)
  S <- within_class_scatter(Z, y)

  B <- (1 - alpha) * M - alpha * S

  eig <- eigen(B, symmetric = TRUE)
  U <- eig$vectors[, seq_len(ncomp), drop = FALSE]

  V <- if (p > n) P %*% U else U
  s <- Xp %*% V

  multivarious::discriminant_projector(
    v       = V,
    s       = s,
    sdev    = apply(s, 2, sd),
    preproc = procres,
    labels  = y,
    method  = "dtrda",
    call    = cl,
    classes = "dtrda"
  )
}
