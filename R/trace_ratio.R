#' Trace Ratio Optimization Returning a Discriminant Projector
#'
#' This function performs trace ratio optimization, a technique commonly used in
#' dimensionality reduction and feature extraction problems such as those found in
#' discriminant analysis. The trace ratio criterion seeks a subspace that maximizes
#' the ratio of the trace of one scatter matrix (\code{A}) to another (\code{B}).
#'
#' Formally, given two symmetric matrices \eqn{A} and \eqn{B}, we want to find a
#' projection matrix \eqn{V} (with orthonormal columns) that maximizes:
#' \deqn{\mathrm{trace}(V^T A V) / \mathrm{trace}(V^T B V).}
#' This is solved iteratively using eigen decomposition methods.
#'
#' **What if you want a \code{discriminant_projector} for typical usage?**  
#' If you provide \code{X} (the original data) and optional \code{y} (class labels),
#' this function will build a \code{discriminant_projector} object, storing:
#' \itemize{
#'   \item \code{v} ~ the \eqn{n x p} projection vectors (the subspace),
#'   \item \code{s} ~ the training scores (\eqn{X \times v}),
#'   \item \code{sdev} ~ standard deviations of each dimension in \code{s},
#'   \item \code{labels} ~ set to \code{y} if provided,
#'   \item \code{classes} ~ set to \code{"trace_ratio"}.
#' }
#' If \code{X} is \code{NULL}, we just return a projector with \code{v} but no training scores.
#'
#' @param A A symmetric numeric matrix representing the "numerator" scatter matrix.
#' @param B A symmetric numeric matrix representing the "denominator" scatter matrix.
#' @param X (optional) A numeric matrix \eqn{(n \times d)} of the original data. 
#'   If provided, we compute the training scores \eqn{X \times V} in the final projector.
#' @param y (optional) A vector of length \eqn{n} of class labels. If provided, stored in
#'   the projector's \code{labels} field.
#' @param ncomp An integer specifying the number of components (dimension of the
#'   subspace) to extract. Default is 2.
#' @param eps A numeric tolerance for convergence. The iterative procedure stops if
#'   the change in the trace ratio (\eqn{\rho}) is less than \code{eps}. Default is 1e-6.
#' @param maxiter An integer specifying the maximum number of iterations for the
#'   optimization. Default is 100.
#'
#' @return A \code{\link[multivarious]{discriminant_projector}} object, with:
#' \itemize{
#'   \item \code{v}: A matrix whose columns are the \eqn{ncomp} vectors
#'   corresponding to the subspace that maximizes the trace ratio criterion.
#'   \item \code{s}: If \code{X} is given, the training scores (\eqn{X \times v}).
#'   \item \code{sdev}: The standard deviations of columns in \code{s}.
#'   \item \code{labels}: set to \code{y} if provided, otherwise \code{NULL}.
#'   \item \code{classes}: A character string \code{"trace_ratio"}.
#' }
#'
#' @details
#' This function solves a generalized eigenvalue-like problem iteratively, updating
#' the projection \eqn{V} until convergence. It uses \code{PRIMME::eigs_sym} for
#' eigen decompositions at each step. The approach is inspired by the trace ratio
#' criterion found in linear discriminant analysis and related dimension reduction
#' techniques.
#'
#' **Algorithm**:
#' \enumerate{
#'   \item Initialize \eqn{V} randomly (size \eqn{n \times ncomp}).
#'   \item At iteration \eqn{t}:
#'     \itemize{
#'       \item Compute \eqn{\rho_t = trace(V^T A V) / trace(V^T B V)}.
#'       \item Form \eqn{M = A - \rho_t B}.
#'       \item Eigen-decompose \eqn{M} to get top \eqn{ncomp} eigenvectors => new \eqn{V}.
#'     }
#'   \item Repeat until \eqn{|\rho_{t+1} - \rho_t| < eps} or maxiter is reached.
#' }
#'
#' @seealso \code{\link[PRIMME]{eigs_sym}} for the eigen decomposition solver,
#' and \code{\link{between_class_scatter}}, \code{\link{within_class_scatter}}
#' for common scatter matrices used in discriminant analysis.
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- scale(iris[,1:4])
#' y <- iris[,5]
#'
#' # Build scatter matrices (a typical case in LDA)
#' A <- between_class_scatter(X, y)
#' B <- within_class_scatter(X, y)
#'
#' # Solve trace ratio with 3 components, storing training scores
#' proj <- trace_ratio(A, B, X = X, y = y, ncomp = 3)
#' print(proj)
#'
#' # You can now project new data with project(proj, newdata), etc.
#' }
#'
#' @export
trace_ratio <- function(A, B,
                        X     = NULL,
                        y     = NULL,
                        ncomp = 2,
                        eps   = 1e-6,
                        maxiter = 100) {
  if (!requireNamespace("PRIMME", quietly = TRUE)) {
    stop("Package 'PRIMME' is required but not installed.")
  }
  if (!is.matrix(A) || !is.matrix(B)) {
    stop("A and B must be numeric matrices.")
  }
  if (nrow(A) != ncol(A) || nrow(B) != ncol(B) || nrow(A) != nrow(B)) {
    stop("A and B must be square symmetric matrices of the same dimension.")
  }
  n <- nrow(A)
  p <- ncomp
  if (p < 1 || p > n) {
    stop("ncomp must be between 1 and n (the dimension of A,B).")
  }
  # Initialize V
  set.seed(123)  # or remove if you want non-deterministic
  V_old <- qr.Q(qr(matrix(rnorm(n * p), nrow=n, ncol=p)))
  
  rho_old <- 0
  for (iter in seq_len(maxiter)) {
    # Build M = A - rho_old * B
    M <- A - rho_old * B
    
    # Eigen decomposition to get top p vectors
    # Possibly add x0 = V_old after a few iters for speed
    if (iter > 3) {
      res_eigs <- PRIMME::eigs_sym(M, k = p, x0 = V_old)
    } else {
      res_eigs <- PRIMME::eigs_sym(M, k = p)
    }
    V_new <- res_eigs$vectors
    
    # compute new ratio
    num_mat <- crossprod(V_new, A) %*% V_new
    den_mat <- crossprod(V_new, B) %*% V_new
    # sum of diag(num_mat)/diag(den_mat) is typical multi-vector extension
    rho_new <- sum(diag(num_mat) / diag(den_mat))
    
    delta <- abs(rho_new - rho_old)
    
    V_old  <- V_new
    rho_old <- rho_new
    
    if (delta < eps) {
      break
    }
  }
  
  # final vectors
  V <- V_old
  
  # build s, sdev, etc. if X is provided
  s <- NULL
  sdev <- rep(NA, ncomp)
  if (!is.null(X)) {
    # project
    if (!is.matrix(X)) {
      stop("'X' must be a matrix if provided.")
    }
    s_mat <- X %*% V
    sdev  <- apply(s_mat, 2, sd)
    s     <- s_mat
  }
  
  # build a discriminant_projector
  # We'll assume we have 'multivarious::discriminant_projector' in your environment
  dp_obj <- multivarious::discriminant_projector(
    v       = V,
    s       = if (!is.null(s)) s else matrix(0, nrow=0, ncol=ncomp),
    sdev    = sdev,
    preproc = multivarious::prep(pass()),  # or pass() as no-op
    labels  = y,
    classes = "trace_ratio"
  )
  
  dp_obj
}