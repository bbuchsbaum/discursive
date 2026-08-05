#' Sparse or Ridge Low-Rank Regression
#'
#' Fits the low-rank ridge regression (LRRR) or sparse low-rank regression
#' (SLRR) models of Cai et al. (2013). Both models regress a normalized class
#' indicator matrix on a low-dimensional discriminant subspace. The sparse
#' variant uses iterative reweighting to impose a row-wise \eqn{\ell_{2,1}}
#' penalty on the coefficient matrix.
#'
#' @param X Numeric matrix with samples in rows and features in columns.
#' @param Y Class labels with length equal to `nrow(X)`.
#' @param s Positive integer rank. It cannot exceed `ncol(X)` or the number of
#'   classes minus one.
#' @param lambda Nonnegative regularization parameter.
#' @param penalty Either `"ridge"` for LRRR or `"l21"` for SLRR.
#' @param max_iter Positive integer giving the maximum number of SLRR updates.
#' @param tol Positive relative convergence tolerance for SLRR.
#' @param preproc A fitted or unfitted preprocessing specification from
#'   \pkg{multivarious}. The default centers the predictors, as required by
#'   the regression formulation.
#' @param verbose Logical; report the SLRR objective after every iteration.
#'
#' @return A [multivarious::discriminant_projector()] with classes `"slrr"`
#'   and the selected penalty. In addition to the standard projector fields it
#'   contains:
#'   \describe{
#'     \item{A}{The `ncol(X)` by `s` discriminant basis.}
#'     \item{B}{The `s` by `nlevels(Y)` regression coefficient matrix.}
#'     \item{eigenvalues}{Generalized eigenvalues associated with `A`.}
#'     \item{objective_history}{The ridge objective, or the SLRR objective at
#'       every reweighting iteration.}
#'     \item{iterations}{Number of optimization steps performed.}
#'     \item{converged}{Whether the relative SLRR stopping rule was met.}
#'   }
#'
#' @details
#' The normalized class indicator is `1 / sqrt(n_j)` for observations in class
#' `j`. For LRRR, `A` contains the leading generalized eigenvectors of
#' \eqn{S_b A = (S_t + \lambda I) A \Lambda}. For SLRR, the identity is
#' replaced at every iteration by the diagonal reweighting matrix whose entries
#' are \eqn{1 / (2 ||w_i||_2)}.
#'
#' The generalized eigenproblem is solved through a symmetric whitening
#' transformation. This avoids treating the generally nonsymmetric product
#' \eqn{(S_t + \lambda D)^{-1} S_b} as a symmetric matrix and also supports
#' semidefinite denominator matrices when their numerical rank is sufficient.
#'
#' @references
#' Cai, X., Ding, C., Nie, F., and Huang, H. (2013). On the Equivalent of
#' Low-rank Linear Regressions and Linear Discriminant Analysis Based
#' Regressions. *Proceedings of KDD '13*, 1124--1132.
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' Y <- iris$Species
#' fit_lrrr <- slrr(X, Y, s = 2, lambda = 0.01, penalty = "ridge")
#' fit_slrr <- slrr(X, Y, s = 2, lambda = 0.01, penalty = "l21")
#'
#' @export
slrr <- function(X,
                 Y,
                 s,
                 lambda = 0.001,
                 penalty = c("ridge", "l21"),
                 max_iter = 50,
                 tol = 1e-6,
                 preproc = center(),
                 verbose = FALSE) {
  penalty <- match.arg(penalty)
  X <- as.matrix(X)

  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (length(Y) != nrow(X) || anyNA(Y)) {
    stop("'Y' must contain one non-missing label per row of 'X'.")
  }

  Y <- droplevels(as.factor(Y))
  n_classes <- nlevels(Y)
  if (n_classes < 2L) {
    stop("'Y' must contain at least two classes.")
  }

  max_rank <- min(ncol(X), n_classes - 1L)
  s <- .check_whole_number(s, "s", lower = 1L, upper = max_rank)
  lambda <- .check_finite_scalar(lambda, "lambda", lower = 0)
  max_iter <- .check_whole_number(max_iter, "max_iter", lower = 1L)
  tol <- .check_finite_scalar(tol, "tol", lower = 0, lower_open = TRUE)
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("'verbose' must be TRUE or FALSE.")
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  # Cai et al. define Y_ij = 1 / sqrt(n_j) for members of class j.
  G <- stats::model.matrix(~ Y - 1)
  G <- sweep(G, 2L, sqrt(colSums(G)), FUN = "/")

  XtX <- crossprod(Xp)
  XtG <- crossprod(Xp, G)
  Sb <- tcrossprod(XtG)
  d <- ncol(Xp)

  solve_step <- function(d_weights) {
    regularized_scatter <- XtX
    diag(regularized_scatter) <- diag(regularized_scatter) + lambda * d_weights

    eig <- .symmetric_generalized_eigen(
      numerator = Sb,
      denominator = regularized_scatter,
      ncomp = s
    )
    A <- eig$vectors
    lhs <- crossprod(A, regularized_scatter %*% A)
    rhs <- crossprod(A, XtG)
    B <- solve(lhs, rhs)
    W <- A %*% B

    list(
      A = A,
      B = B,
      W = W,
      eigenvalues = eig$values,
      regularized_scatter = regularized_scatter
    )
  }

  if (penalty == "ridge") {
    solution <- solve_step(rep(1, d))
    residual <- G - Xp %*% solution$W
    objective_history <- sum(residual^2) + lambda * sum(solution$W^2)
    iterations <- 1L
    converged <- TRUE
  } else {
    d_weights <- rep(1, d)
    objective_history <- numeric(max_iter)
    converged <- FALSE

    for (iter in seq_len(max_iter)) {
      solution <- solve_step(d_weights)
      row_norms <- sqrt(rowSums(solution$W^2))
      residual <- G - Xp %*% solution$W
      objective_history[iter] <- sum(residual^2) + lambda * sum(row_norms)

      if (verbose) {
        message(sprintf("Iter=%d, objective=%.10g", iter, objective_history[iter]))
      }

      if (iter > 1L) {
        change <- abs(objective_history[iter] - objective_history[iter - 1L])
        scale <- 1 + abs(objective_history[iter - 1L])
        if (change <= tol * scale) {
          converged <- TRUE
          break
        }
      }

      d_weights <- 1 / (2 * pmax(row_norms, sqrt(.Machine$double.eps)))
    }

    iterations <- iter
    objective_history <- objective_history[seq_len(iterations)]
  }

  scores <- Xp %*% solution$W
  multivarious::discriminant_projector(
    v = solution$W,
    s = scores,
    sdev = apply(scores, 2L, stats::sd),
    preproc = procres,
    labels = Y,
    classes = c("slrr", penalty),
    A = solution$A,
    B = solution$B,
    eigenvalues = solution$eigenvalues,
    objective_history = objective_history,
    iterations = iterations,
    converged = converged
  )
}

.symmetric_generalized_eigen <- function(numerator, denominator, ncomp) {
  numerator <- 0.5 * (numerator + t(numerator))
  denominator <- 0.5 * (denominator + t(denominator))

  denominator_eig <- eigen(denominator, symmetric = TRUE)
  spectral_scale <- max(1, max(abs(denominator_eig$values)))
  keep <- denominator_eig$values > sqrt(.Machine$double.eps) * spectral_scale
  if (sum(keep) < ncomp) {
    stop("The regularized scatter matrix has insufficient numerical rank for 's'.")
  }

  whitening <- sweep(
    denominator_eig$vectors[, keep, drop = FALSE],
    2L,
    sqrt(denominator_eig$values[keep]),
    FUN = "/"
  )
  reduced <- crossprod(whitening, numerator %*% whitening)
  reduced <- 0.5 * (reduced + t(reduced))
  reduced_eig <- eigen(reduced, symmetric = TRUE)
  take <- seq_len(ncomp)

  list(
    values = reduced_eig$values[take],
    vectors = whitening %*% reduced_eig$vectors[, take, drop = FALSE]
  )
}

.check_finite_scalar <- function(x, name, lower = -Inf, upper = Inf,
                                 lower_open = FALSE, upper_open = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    stop(sprintf("'%s' must be a finite numeric scalar.", name))
  }
  lower_bad <- if (lower_open) x <= lower else x < lower
  upper_bad <- if (upper_open) x >= upper else x > upper
  if (lower_bad || upper_bad) {
    interval <- paste0(
      if (lower_open) "(" else "[", lower, ", ", upper,
      if (upper_open) ")" else "]"
    )
    stop(sprintf("'%s' must lie in %s.", name, interval))
  }
  as.numeric(x)
}

.check_whole_number <- function(x, name, lower = -Inf, upper = Inf) {
  x <- .check_finite_scalar(x, name, lower = lower, upper = upper)
  if (x != floor(x)) {
    stop(sprintf("'%s' must be an integer.", name))
  }
  as.integer(x)
}
