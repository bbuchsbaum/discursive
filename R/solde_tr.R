#' Stable Orthogonal Local Discriminant Embedding by Trace Ratio
#'
#' Fits SOLDE-TR using sparse nearest-neighbor graphs and a deterministic
#' iterative trace-ratio solver. Predictors are always centered. Optionally,
#' PCA is used to remove null directions before graph construction and
#' optimization.
#'
#' @param X Numeric matrix with samples in rows and features in columns.
#' @param y Class labels with length equal to `nrow(X)`.
#' @param s Positive integer number of nearest neighbors.
#' @param alpha Number in `[0.5, 1]` controlling the similarity/diversity
#'   balance in \eqn{L_d = \alpha L_s - (1 - \alpha)L_v}.
#' @param sigma Positive heat-kernel scale in
#'   \eqn{\exp(-||x_i-x_j||^2 / \sigma)}.
#' @param m Positive integer number of projection dimensions.
#' @param tol Positive relative convergence tolerance for the trace ratio.
#' @param maxit Positive integer maximum number of trace-ratio iterations.
#' @param pca_preprocess Logical; remove numerically null PCA directions before
#'   fitting.
#' @param denominator_tol Positive relative eigenvalue floor for the
#'   trace-ratio denominator. If the constructed denominator is indefinite or
#'   nearly singular, the smallest diagonal shift that reaches this floor is
#'   applied and returned in `denominator_shift`.
#'
#' @return A [multivarious::discriminant_projector()] with class `"solde_tr"`.
#'   In addition to the standard projector fields it contains:
#'   \describe{
#'     \item{W_pca}{PCA loading matrix, or a sparse identity matrix when PCA
#'       is disabled.}
#'     \item{W_solde_tr}{Orthogonal trace-ratio directions in the fitted
#'       coordinate system.}
#'     \item{eigvals_pca}{Retained PCA eigenvalues, or `NULL`.}
#'     \item{trace_ratio_history}{Objective values for accepted iterates.}
#'     \item{iterations}{Number of trace-ratio iterations performed.}
#'     \item{converged}{Whether the stopping rule was met.}
#'     \item{denominator_shift}{Adaptive diagonal regularization applied to the
#'       trace-ratio denominator.}
#'   }
#'
#' @details
#' A single nearest-neighbor query is shared by all three graphs. Self indices
#' are removed explicitly, which remains correct under zero-distance ties. The
#' similarity (`S`), diversity (`H`), and interclass (`F`) graphs and their
#' Laplacians are sparse, so storage grows with the number of neighbor edges
#' rather than quadratically in the number of samples.
#'
#' The trace-ratio iteration maximizes
#' \deqn{tr(W^T S_p W) / tr(W^T S_d W), \quad W^T W = I.}
#' At each iteration it extracts the leading eigenvectors of
#' \eqn{S_p - \rho S_d}. The initial ratio is zero, making the procedure
#' deterministic and leaving R's random-number state unchanged.
#'
#' @references
#' Yang, X., Liu, G., Yu, Q., and Wang, R. (2018). Stable and orthogonal local
#' discriminant embedding using trace ratio criterion for dimensionality
#' reduction. *Multimedia Tools and Applications*, 77, 3071--3081.
#' \doi{10.1007/s11042-017-5022-1}
#'
#' Wang, H., Yan, S., Xu, D., Tang, X., and Huang, T. (2007). Trace ratio vs.
#' ratio trace for dimensionality reduction. *Proceedings of CVPR 2007*.
#' \doi{10.1109/CVPR.2007.382983}
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 8), nrow = 60)
#' y <- factor(rep(1:3, each = 20))
#' fit <- solde_tr(X, y, s = 4, m = 2)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
#'
#' @export
solde_tr <- function(X,
                     y,
                     s = 5,
                     alpha = 0.9,
                     sigma = 1,
                     m = 10,
                     tol = 1e-6,
                     maxit = 100,
                     pca_preprocess = TRUE,
                     denominator_tol = 1e-8) {
  if (!requireNamespace("Rnanoflann", quietly = TRUE)) {
    stop("Package 'Rnanoflann' is required by 'solde_tr'.")
  }

  X <- as.matrix(X)
  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (length(y) != nrow(X) || anyNA(y)) {
    stop("'y' must contain one non-missing label per row of 'X'.")
  }

  y <- droplevels(as.factor(y))
  if (nlevels(y) < 2L) {
    stop("'y' must contain at least two classes.")
  }

  s <- .check_whole_number(s, "s", lower = 1L, upper = nrow(X) - 1L)
  alpha <- .check_finite_scalar(alpha, "alpha", lower = 0.5, upper = 1)
  sigma <- .check_finite_scalar(sigma, "sigma", lower = 0, lower_open = TRUE)
  m <- .check_whole_number(m, "m", lower = 1L)
  tol <- .check_finite_scalar(tol, "tol", lower = 0, lower_open = TRUE)
  maxit <- .check_whole_number(maxit, "maxit", lower = 1L)
  denominator_tol <- .check_finite_scalar(
    denominator_tol, "denominator_tol", lower = 0, lower_open = TRUE
  )
  if (!is.logical(pca_preprocess) || length(pca_preprocess) != 1L || is.na(pca_preprocess)) {
    stop("'pca_preprocess' must be TRUE or FALSE.")
  }

  n <- nrow(X)
  d <- ncol(X)
  center_fit <- multivarious::fit(multivarious::center(), X)
  X_centered <- as.matrix(multivarious::transform(center_fit, X))

  if (pca_preprocess) {
    max_components <- min(n - 1L, d)
    pca_svd <- svd(X_centered, nu = 0L, nv = max_components)
    singular_scale <- max(1, max(pca_svd$d))
    keep <- pca_svd$d > max(dim(X_centered)) * .Machine$double.eps * singular_scale
    W_pca <- pca_svd$v[, keep, drop = FALSE]
    eigvals_pca <- pca_svd$d[keep]^2 / (n - 1)
    X_fit <- X_centered %*% W_pca
  } else {
    W_pca <- Matrix::Diagonal(d)
    eigvals_pca <- NULL
    X_fit <- X_centered
  }

  fitted_dimension <- ncol(X_fit)
  if (fitted_dimension < m) {
    stop(sprintf(
      "The fitted coordinate system has dimension %d, which is smaller than 'm' = %d.",
      fitted_dimension, m
    ))
  }

  graphs <- .solde_adjacency_graphs(X_fit, y, s = s, sigma = sigma)
  L_s <- Matrix::Diagonal(x = Matrix::colSums(graphs$S)) - graphs$S
  L_v <- Matrix::Diagonal(x = Matrix::colSums(graphs$H)) - graphs$H
  L_m <- Matrix::Diagonal(x = Matrix::colSums(graphs$F)) - graphs$F
  L_d <- alpha * L_s - (1 - alpha) * L_v

  S_p <- crossprod(X_fit, L_m %*% X_fit)
  S_d <- crossprod(X_fit, L_d %*% X_fit)
  S_p <- 0.5 * (S_p + t(S_p))
  S_d <- 0.5 * (S_d + t(S_d))

  denominator_eigenvalues <- eigen(S_d, symmetric = TRUE, only.values = TRUE)$values
  denominator_scale <- max(1, max(abs(denominator_eigenvalues)))
  denominator_floor <- denominator_tol * denominator_scale
  denominator_shift <- max(0, denominator_floor - min(denominator_eigenvalues))
  if (denominator_shift > 10 * denominator_floor) {
    warning(sprintf(
      paste0(
        "The SOLDE-TR denominator was materially indefinite; ",
        "an adaptive diagonal shift of %.6g was applied."
      ),
      denominator_shift
    ), call. = FALSE)
  }
  diag(S_d) <- diag(S_d) + denominator_shift

  trace_fit <- .solde_trace_ratio(
    numerator = S_p,
    denominator = S_d,
    ncomp = m,
    tol = tol,
    maxit = maxit
  )
  W_solde_tr <- trace_fit$vectors
  W_final <- if (pca_preprocess) W_pca %*% W_solde_tr else W_solde_tr
  scores <- X_centered %*% W_final

  multivarious::discriminant_projector(
    v = W_final,
    s = scores,
    sdev = apply(scores, 2L, stats::sd),
    preproc = center_fit,
    labels = y,
    classes = "solde_tr",
    W_pca = W_pca,
    W_solde_tr = W_solde_tr,
    eigvals_pca = eigvals_pca,
    trace_ratio_history = trace_fit$history,
    iterations = trace_fit$iterations,
    converged = trace_fit$converged,
    denominator_shift = denominator_shift,
    denominator_min_eigenvalue = min(denominator_eigenvalues) + denominator_shift,
    graph_edge_counts = c(
      similarity = length(graphs$S@x) / 2,
      diversity = length(graphs$H@x) / 2,
      interclass = length(graphs$F@x) / 2
    )
  )
}

.solde_adjacency_graphs <- function(X, y, s, sigma) {
  n <- nrow(X)
  nn_out <- suppressWarnings(Rnanoflann::nn(
    data = X,
    points = X,
    k = min(s + 1L, n),
    method = "euclidean",
    search = "standard",
    eps = 0,
    square = FALSE,
    trans = TRUE,
    sorted = TRUE
  ))

  from <- vector("list", n)
  to <- vector("list", n)
  distances <- vector("list", n)
  for (i in seq_len(n)) {
    candidate_index <- nn_out$indices[i, ]
    candidate_distance <- nn_out$distances[i, ]
    keep <- which(candidate_index != i & candidate_index >= 1L & candidate_index <= n)
    keep <- head(keep, s)
    if (length(keep) < s) {
      stop("Nearest-neighbor search returned fewer than 's' non-self neighbors.")
    }
    from[[i]] <- rep.int(i, s)
    to[[i]] <- candidate_index[keep]
    distances[[i]] <- candidate_distance[keep]
  }

  from <- unlist(from, use.names = FALSE)
  to <- unlist(to, use.names = FALSE)
  distances <- unlist(distances, use.names = FALSE)
  lo <- pmin(from, to)
  hi <- pmax(from, to)
  pair_key <- lo + (hi - 1) * n
  pair_order <- order(pair_key, distances)
  pair_key <- pair_key[pair_order]
  unique_pair <- !duplicated(pair_key)
  lo <- lo[pair_order][unique_pair]
  hi <- hi[pair_order][unique_pair]
  distances <- distances[pair_order][unique_pair]

  kernel <- exp(-(distances^2) / sigma)
  same_class <- y[lo] == y[hi]
  make_symmetric <- function(select, values) {
    Matrix::drop0(Matrix::sparseMatrix(
      i = c(lo[select], hi[select]),
      j = c(hi[select], lo[select]),
      x = rep(values[select], 2L),
      dims = c(n, n),
      giveCsparse = TRUE
    ))
  }

  list(
    S = make_symmetric(same_class, kernel),
    H = make_symmetric(same_class, 1 - kernel),
    F = make_symmetric(!same_class, kernel)
  )
}

.solde_trace_ratio <- function(numerator, denominator, ncomp, tol, maxit) {
  numerator <- 0.5 * (numerator + t(numerator))
  denominator <- 0.5 * (denominator + t(denominator))
  rho <- 0
  history <- numeric(maxit)
  converged <- FALSE

  for (iter in seq_len(maxit)) {
    difference <- numerator - rho * denominator
    eig <- .top_symmetric_eigen(difference, ncomp)
    vectors <- eig$vectors
    trace_numerator <- sum(vectors * (numerator %*% vectors))
    trace_denominator <- sum(vectors * (denominator %*% vectors))
    if (!is.finite(trace_denominator) || trace_denominator <= 0) {
      stop("The trace-ratio denominator is not positive for the current iterate.")
    }

    rho_new <- trace_numerator / trace_denominator
    history[iter] <- rho_new
    if (iter > 1L && abs(rho_new - rho) <= tol * (1 + abs(rho))) {
      converged <- TRUE
      rho <- rho_new
      break
    }
    rho <- rho_new
  }

  list(
    vectors = vectors,
    value = rho,
    history = history[seq_len(iter)],
    iterations = iter,
    converged = converged
  )
}

.top_symmetric_eigen <- function(x, ncomp) {
  n <- nrow(x)
  if (n <= 100L || ncomp >= n - 1L) {
    eig <- eigen(x, symmetric = TRUE)
    take <- seq_len(ncomp)
    return(list(values = eig$values[take], vectors = eig$vectors[, take, drop = FALSE]))
  }

  initvec <- sin(seq_len(n)) + cos(seq_len(n) * sqrt(2))
  initvec <- initvec / sqrt(sum(initvec^2))
  eig <- tryCatch(
    RSpectra::eigs_sym(
      x,
      k = ncomp,
      which = "LA",
      opts = list(initvec = initvec, tol = 1e-10)
    ),
    error = function(e) NULL
  )
  if (is.null(eig)) {
    eig <- eigen(x, symmetric = TRUE)
    take <- seq_len(ncomp)
    return(list(values = eig$values[take], vectors = eig$vectors[, take, drop = FALSE]))
  }

  ordering <- order(Re(eig$values), decreasing = TRUE)
  list(
    values = Re(eig$values[ordering]),
    vectors = Re(eig$vectors[, ordering, drop = FALSE])
  )
}
