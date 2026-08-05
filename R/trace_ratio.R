#' Trace-Ratio Subspace Optimization
#'
#' Finds an orthonormal basis that maximizes the ratio of traces
#' \deqn{\mathrm{tr}(V^T A V) / \mathrm{tr}(V^T B V).}
#' The returned object is a multivarious discriminant projector so an
#' optional training matrix can be retained and projected coherently.
#'
#' @param A Symmetric finite numeric numerator matrix.
#' @param B Symmetric finite positive-semidefinite denominator matrix. It must
#'   yield a positive trace on every iterate used by the solver.
#' @param X Optional numeric training matrix with `nrow(A)` features.
#' @param y Optional labels, one per row of `X`. When omitted, a single
#'   `"unlabeled"` level is stored because discriminant projectors require one
#'   label per training score.
#' @param ncomp Positive integer number of orthonormal components.
#' @param eps Positive relative convergence tolerance.
#' @param maxiter Positive integer maximum number of trace-ratio iterations.
#'
#' @return A [multivarious::discriminant_projector()] with class
#'   `"trace_ratio"`. Additional fields include `value`, `history`,
#'   `iterations`, and `converged`.
#'
#' @details
#' At each iteration the solver takes the leading eigenvectors of
#' \eqn{A-\rho B} and updates
#' \deqn{\rho = \frac{\sum_j v_j^T A v_j}
#'                    {\sum_j v_j^T B v_j}.}
#' This is a ratio of traces. It is not the sum of the componentwise Rayleigh
#' quotients. The eigensolver and initialization are deterministic and this
#' function neither reads from nor changes R's random-number stream.
#'
#' @references
#' Wang, H., Yan, S., Xu, D., Tang, X., and Huang, T. (2007). Trace ratio vs.
#' ratio trace for dimensionality reduction. *Proceedings of CVPR*, 1--8.
#'
#' @seealso [solde_tr()]
#' @export
#'
#' @examples
#' A <- diag(c(9, 8, 1))
#' B <- diag(c(1, 4, 1))
#' fit <- trace_ratio(A, B, ncomp = 2)
#' fit$value
trace_ratio <- function(A,
                        B,
                        X = NULL,
                        y = NULL,
                        ncomp = 2,
                        eps = 1e-6,
                        maxiter = 100) {
  if (!is.matrix(A) || !is.numeric(A) || !is.matrix(B) || !is.numeric(B)) {
    stop("'A' and 'B' must be numeric matrices.")
  }
  if (nrow(A) != ncol(A) || any(dim(A) != dim(B))) {
    stop("'A' and 'B' must be square matrices with the same dimensions.")
  }
  if (any(!is.finite(A)) || any(!is.finite(B))) {
    stop("'A' and 'B' must contain only finite values.")
  }
  if (!isSymmetric(A) || !isSymmetric(B)) {
    stop("'A' and 'B' must be symmetric.")
  }

  n <- nrow(A)
  ncomp <- .check_whole_number(ncomp, "ncomp", lower = 1L, upper = n)
  eps <- .check_finite_scalar(eps, "eps", lower = 0, lower_open = TRUE)
  maxiter <- .check_whole_number(maxiter, "maxiter", lower = 1L)

  B_eigenvalues <- eigen(B, symmetric = TRUE, only.values = TRUE)$values
  B_scale <- max(1, max(abs(B_eigenvalues)))
  if (min(B_eigenvalues) < -sqrt(.Machine$double.eps) * B_scale) {
    stop("'B' must be positive semidefinite.")
  }
  if (max(B_eigenvalues) <= .Machine$double.eps * B_scale) {
    stop("'B' must have positive numerical rank.")
  }

  fit <- .solde_trace_ratio(
    numerator = A,
    denominator = B,
    ncomp = ncomp,
    tol = eps,
    maxit = maxiter
  )
  vectors <- fit$vectors
  component_names <- paste0("TR", seq_len(ncomp))
  rownames(vectors) <- rownames(A)
  colnames(vectors) <- component_names

  if (is.null(X)) {
    scores <- matrix(numeric(), nrow = 0L, ncol = ncomp)
    colnames(scores) <- component_names
    score_sdev <- rep(NA_real_, ncomp)
    names(score_sdev) <- component_names
    labels <- factor(character())
    preproc_obj <- multivarious::fit(
      multivarious::pass(), matrix(0, nrow = 1L, ncol = n)
    )
  } else {
    X <- as.matrix(X)
    if (!is.numeric(X) || nrow(X) < 2L || ncol(X) != n || any(!is.finite(X))) {
      stop("'X' must be a finite numeric matrix with at least two rows and nrow(A) columns.")
    }
    if (!is.null(y) && (length(y) != nrow(X) || anyNA(y))) {
      stop("'y' must contain one non-missing label per row of 'X'.")
    }
    labels <- if (is.null(y)) {
      factor(rep("unlabeled", nrow(X)))
    } else {
      droplevels(factor(y))
    }
    scores <- X %*% vectors
    colnames(scores) <- component_names
    score_sdev <- apply(scores, 2L, stats::sd)
    names(score_sdev) <- component_names
    preproc_obj <- multivarious::fit(multivarious::pass(), X)
  }

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = preproc_obj,
    labels = labels,
    classes = "trace_ratio",
    value = fit$value,
    history = fit$history,
    iterations = fit$iterations,
    converged = fit$converged
  )
}
