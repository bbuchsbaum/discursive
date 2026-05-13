#' Orthogonal Linear Discriminant Analysis
#'
#' Performs Orthogonal Linear Discriminant Analysis (OLDA). OLDA finds
#' orthogonal linear discriminants that maximise between-class separation. It
#' is useful for dimensionality reduction and classification tasks.
#'
#' @param X A numeric matrix of size `n x d` (rows = samples, columns = features).
#' @param y A factor or numeric vector of class labels of length `n`. If
#'   numeric, it is coerced to a factor.
#' @param ncomp Integer. Number of orthogonal discriminant directions to
#'   retain. If `NULL` (default), all directions returned by [ulda()] are kept.
#' @param preproc A preprocessing function from `multivarious`. Defaults to
#'   [multivarious::center()].
#'
#' @return A `discriminant_projector` object (subclass `"olda"`) — see
#'   [print.discriminant_projector()] for the standard display.
#' @export
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris[, 5]
#' res <- olda(X, y)
#' print(res)
#' }
#'
#' @seealso [ulda()], [multivarious::discriminant_projector()]
olda <- function(X, y, ncomp = NULL, preproc = multivarious::center()) {
  cl <- match.call()
  if (!is.matrix(X)) X <- as.matrix(X)
  if (nrow(X) != length(y)) {
    stop("'y' must have the same number of observations as rows in 'X'.")
  }
  y <- as.factor(y)

  res_lda <- ulda(X, y, preproc = preproc)
  v_init  <- coef(res_lda)
  v_ortho <- qr.Q(qr(v_init))
  if (!is.null(ncomp)) {
    ncomp <- as.integer(ncomp)
    if (ncomp < 1 || ncomp > ncol(v_ortho)) {
      stop(sprintf("'ncomp' must be between 1 and %d.", ncol(v_ortho)))
    }
    v_ortho <- v_ortho[, seq_len(ncomp), drop = FALSE]
  }

  preproc_final <- res_lda$preproc
  Xp <- multivarious::init_transform(preproc_final, X)
  s_mat <- Xp %*% v_ortho

  multivarious::discriminant_projector(
    v       = v_ortho,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = preproc_final,
    labels  = y,
    method  = "olda",
    call    = cl,
    classes = "olda"
  )
}
