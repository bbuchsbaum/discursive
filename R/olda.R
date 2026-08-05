#' Orthogonal Linear Discriminant Analysis
#'
#' Fits the orthogonal linear discriminant analysis (OLDA) solution of Ye
#' (2005). OLDA returns an orthonormal basis for the discriminant subspace
#' found by uncorrelated linear discriminant analysis (ULDA).
#'
#' @param X Numeric matrix or data frame with samples in rows and features in
#'   columns.
#' @param Y Class labels with length equal to `nrow(X)`.
#' @param preproc A fitted or unfitted preprocessing specification from
#'   \pkg{multivarious}. The default centers the predictors.
#' @param mu Nonnegative compatibility parameter passed to [ulda()]. The ULDA
#'   SVD solution is independent of this value.
#' @param tol Relative numerical-rank tolerance in `(0, 1)`, passed to [ulda()].
#' @param ncomp Optional positive integer giving the number of discriminant
#'   components. The default, `NULL`, uses the numerical between-class rank.
#'
#' @return A [multivarious::discriminant_projector()] with class `"olda"`.
#'   In addition to the standard projector fields, it contains:
#'   \describe{
#'     \item{qr_R}{The upper-triangular factor relating the returned OLDA
#'       basis to the pivoted ULDA basis.}
#'     \item{ulda_pivot}{The ULDA component permutation used by the pivoted QR
#'       decomposition.}
#'     \item{ulda_eigenvalues}{The generalized eigenvalues of the source ULDA
#'       solution, in their original discriminant order.}
#'     \item{ncomp}{The number of returned components.}
#'     \item{total_rank}{The numerical rank of the total scatter matrix.}
#'     \item{tol}{The relative rank tolerance used by the fit.}
#'     \item{mu}{The compatibility value supplied to `mu`.}
#'   }
#'
#' @details
#' If \eqn{G_U} is the ULDA basis, OLDA computes a column-pivoted QR
#' decomposition
#' \deqn{G_U P = Q R}
#' and returns \eqn{Q}. Thus \eqn{Q^T Q = I}, while \eqn{Q} spans the same
#' discriminant subspace as \eqn{G_U}. This is Algorithm 2 of Ye (2005).
#'
#' Unlike ULDA, OLDA guarantees Euclidean orthogonality of the loading vectors;
#' it does not generally make the score columns uncorrelated. The QR pivot and
#' triangular factor are retained so the transformation from the source ULDA
#' basis is auditable. Training scores are obtained from the existing ULDA
#' scores with a triangular solve, avoiding a second preprocessing and matrix
#' multiplication pass.
#'
#' @references
#' Ye, J. (2005). Characterization of a Family of Algorithms for Generalized
#' Discriminant Analysis on Undersampled Problems. *Journal of Machine Learning
#' Research*, 6, 483--502.
#'
#' @seealso [ulda()], [fastolda()],
#'   [multivarious::discriminant_projector()]
#'
#' @importFrom multivarious center
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' fit <- olda(X, iris$Species, ncomp = 2)
#' crossprod(coef(fit))
#' multivarious::project(fit, X[1:5, , drop = FALSE])
olda <- function(X,
                 Y,
                 preproc = center(),
                 mu = 0,
                 tol = 1e-6,
                 ncomp = NULL) {
  source <- ulda(
    X = X,
    Y = Y,
    preproc = preproc,
    mu = mu,
    tol = tol,
    ncomp = ncomp
  )

  source_vectors <- source$v
  source_ncomp <- ncol(source_vectors)
  decomposition <- qr(source_vectors, LAPACK = TRUE)
  vectors <- qr.Q(decomposition, complete = FALSE)
  triangular <- qr.R(decomposition, complete = FALSE)
  pivot <- decomposition$pivot[seq_len(source_ncomp)]

  # Resolve arbitrary QR signs while preserving source_vectors[, pivot] = Q R.
  signs <- vapply(seq_len(source_ncomp), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")
  triangular <- sweep(triangular, 1L, signs, FUN = "*")

  component_names <- paste0("OLDA", seq_len(source_ncomp))
  source_names <- colnames(source_vectors)[pivot]
  rownames(vectors) <- rownames(source_vectors)
  colnames(vectors) <- component_names
  dimnames(triangular) <- list(component_names, source_names)

  # source$s[, pivot] = scores %*% triangular. Solving this relation avoids
  # transforming X and multiplying by the new basis a second time.
  scores <- t(backsolve(
    triangular,
    t(source$s[, pivot, drop = FALSE]),
    transpose = TRUE
  ))
  colnames(scores) <- component_names
  centered_scores <- sweep(scores, 2L, colMeans(scores), FUN = "-")
  score_sdev <- sqrt(colSums(centered_scores^2) / (nrow(scores) - 1L))
  names(score_sdev) <- component_names

  source_eigenvalues <- source$eigenvalues
  names(source_eigenvalues) <- colnames(source_vectors)

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = source$preproc,
    labels = source$labels,
    classes = "olda",
    qr_R = triangular,
    ulda_pivot = pivot,
    ulda_eigenvalues = source_eigenvalues,
    ncomp = source_ncomp,
    total_rank = source$total_rank,
    tol = source$tol,
    mu = source$mu
  )
}
