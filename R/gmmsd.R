#' Generalized Multiple Maximum Scatter Difference (GMMSD)
#'
#' Solves a symmetric generalised eigenvalue problem to find a projection that
#' maximises the difference between the between-class scatter and a scaled
#' within-class scatter. A QR decomposition is used for efficiency on
#' high-dimensional data.
#'
#' @param X A numeric matrix (`n x d`).
#' @param y A factor or numeric vector of length `n` of class labels. If
#'   numeric, it is coerced to a factor.
#' @param ncomp Integer. Number of dimensions to retain in the transformed
#'   feature space. Default 2.
#' @param c Numeric balance parameter scaling the within-class scatter matrix.
#'   Typically positive. Default 1.
#' @param preproc A `pre_processor` from `multivarious`. Defaults to
#'   [multivarious::center()].
#'
#' @return A `discriminant_projector` object (subclass `"gmmsd"`).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, -5])
#' y <- iris$Species
#' dp <- gmmsd(X, y, ncomp = 2, c = 1)
#' print(dp)
#' }
gmmsd <- function(X, y, ncomp = 2, c = 1,
                  preproc = multivarious::center()) {

  cl <- match.call()
  if (!is.factor(y)) y <- factor(y)

  procres <- multivarious::prep(preproc)
  Xp <- multivarious::init_transform(procres, X)

  qr.decomp <- qr(t(Xp))
  Q1 <- qr.Q(qr.decomp)

  Sb <- between_class_scatter(Xp, y)
  Sw <- within_class_scatter(Xp, y)

  M <- t(Q1) %*% (Sb - c * Sw) %*% Q1

  eigres <- eigen(M, symmetric = TRUE)
  if (ncomp > ncol(Q1)) {
    stop("'ncomp' exceeds available dimensions")
  }
  W <- Q1 %*% eigres$vectors[, seq_len(ncomp), drop = FALSE]

  s <- Xp %*% W

  multivarious::discriminant_projector(
    v       = W,
    s       = s,
    sdev    = apply(s, 2, sd),
    preproc = procres,
    labels  = y,
    method  = "gmmsd",
    call    = cl,
    classes = "gmmsd"
  )
}
