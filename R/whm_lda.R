#' Weighted Harmonic Mean of Trace Ratios Discriminant Analysis
#'
#' Fits the weighted harmonic mean of trace ratios (WHM-TR) method of Li et
#' al. (2017). WHM-TR emphasizes poorly separated class pairs by minimizing a
#' weighted sum of pairwise within- to between-class trace ratios.
#'
#' @param X Numeric matrix or data frame with samples in rows and features in
#'   columns. It must contain at least two columns and only finite values.
#' @param labels A non-missing vector with one class label per row of `X` and
#'   at least two observed classes. Unused factor levels are dropped.
#' @param m Positive integer number of projection dimensions. It must be less
#'   than `ncol(X)`.
#' @param max_iter Positive integer maximum number of WHM-TR updates.
#' @param tol Convergence tolerance in `(0, 1)`. Convergence is measured by the
#'   Frobenius distance between successive projection matrices, so it is
#'   invariant to sign changes and rotations of the fitted basis.
#' @param initial_W Optional finite `ncol(X)` by `m` full-rank initialization.
#'   Its column space is retained and its columns are orthonormalized. The
#'   default uses a random orthonormal basis.
#' @param seed Optional nonnegative integer used for random initialization.
#'   The caller's random-number-generator state is restored after fitting.
#'   It must be `NULL` when `initial_W` is supplied.
#' @param denominator_tol Relative tolerance in `(0, 1)` for projected
#'   pairwise between-class traces. An iteration is stopped with an informative
#'   error when a denominator is not finite or is numerically zero relative to
#'   the corresponding full-space between-class trace.
#'
#' @return A [multivarious::discriminant_projector()] with class `"whm_lda"`.
#'   In addition to the standard projector fields, it contains:
#'   \describe{
#'     \item{objective, objective_history}{The final WHM-TR objective and its
#'       values at initialization and after every update.}
#'     \item{iterations, converged, projector_change}{Iteration count,
#'       convergence flag, and final subspace change.}
#'     \item{stationarity_residual}{Relative KKT residual of the final
#'       subspace.}
#'     \item{class_counts, pair_classes, pair_weights}{Observed class sizes,
#'       class-pair labels, and the paper's `n_j + n_k` pair weights.}
#'     \item{pair_denominators, pair_objectives}{Final projected between-class
#'       traces and weighted pairwise objective terms.}
#'     \item{m, max_iter, tol, denominator_tol, seed}{Fitted dimensions and
#'       solver controls.}
#'   }
#'
#' @details
#' For classes `j` and `k`, let `S_w^{jk}` be their combined within-class
#' scatter, let `mu_j` and `mu_k` be their means, and define
#' \deqn{S_b^{jk} = \frac{n_j n_k}{n_j + n_k}
#'   (\mu_j-\mu_k)(\mu_j-\mu_k)^T.}
#' The fitted orthonormal matrix `W` minimizes
#' \deqn{\sum_{j<k} (n_j+n_k)
#'   \frac{\mathrm{tr}(W^T S_w^{jk} W)}
#'        {\mathrm{tr}(W^T S_b^{jk} W)}}
#' subject to `W^T W = I`. Each update constructs Equation 31 of Li et al.
#' (2017) and selects its `m` smallest-eigenvalue eigenvectors.
#'
#' Class means, counts, and within-class scatters are computed once. Each
#' iteration then uses cached scatter matrices and the rank-one form of every
#' between-class scatter.
#'
#' @references
#' Li, Z., Nie, F., Chang, X., & Yang, Y. (2017). Beyond Trace Ratio:
#' Weighted Harmonic Mean of Trace Ratios for Multiclass Discriminant
#' Analysis. *IEEE Transactions on Knowledge and Data Engineering*, 29(10),
#' 2100--2110. \doi{10.1109/TKDE.2017.2728531}
#'
#' @seealso [multivarious::discriminant_projector()]
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' fit <- whm_lda(X, iris$Species, m = 2, seed = 1)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
whm_lda <- function(X,
                    labels,
                    m,
                    max_iter = 100L,
                    tol = 1e-6,
                    initial_W = NULL,
                    seed = NULL,
                    denominator_tol = 1e-12) {
  X <- as.matrix(X)
  if (!is.numeric(X) || length(dim(X)) != 2L ||
      nrow(X) < 2L || ncol(X) < 2L) {
    stop("'X' must be a numeric matrix with at least two rows and two columns.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (!is.atomic(labels) || !is.null(dim(labels)) ||
      length(labels) != nrow(X)) {
    stop("'labels' must be a vector with one value per row of 'X'.")
  }
  if (anyNA(labels)) {
    stop("'labels' must not contain missing values.")
  }

  labels <- droplevels(as.factor(labels))
  class_levels <- levels(labels)
  n_classes <- length(class_levels)
  if (n_classes < 2L) {
    stop("'labels' must contain at least two observed classes.")
  }

  d <- ncol(X)
  m <- .check_whole_number(m, "m", lower = 1L, upper = d - 1L)
  max_iter <- .check_whole_number(max_iter, "max_iter", lower = 1L)
  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  denominator_tol <- .check_finite_scalar(
    denominator_tol, "denominator_tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  if (!is.null(seed)) {
    seed <- .check_whole_number(
      seed, "seed", lower = 0L, upper = .Machine$integer.max
    )
  }
  if (!is.null(initial_W) && !is.null(seed)) {
    stop("'seed' must be NULL when 'initial_W' is supplied.")
  }

  statistics <- .whm_prepare_statistics(X, labels, class_levels)
  W <- .whm_initialize(d, m, initial_W, seed)

  objective_history <- numeric(max_iter + 1L)
  current <- .whm_evaluate(W, statistics, denominator_tol, need_matrix = TRUE)
  objective_history[1L] <- current$objective
  converged <- FALSE
  projector_change <- Inf

  for (iter in seq_len(max_iter)) {
    update <- .whm_bottom_eigen(current$matrix, m)
    W_new <- update$vectors
    projector_change <- norm(
      tcrossprod(W) - tcrossprod(W_new), type = "F"
    )
    next_state <- .whm_evaluate(
      W_new, statistics, denominator_tol, need_matrix = TRUE
    )
    objective_history[iter + 1L] <- next_state$objective

    W <- W_new
    current <- next_state
    if (projector_change <= tol * sqrt(m)) {
      converged <- TRUE
      break
    }
  }

  objective_history <- objective_history[seq_len(iter + 1L)]
  if (!converged) {
    warning(sprintf(
      "WHM-TR did not converge after %d iterations; returning the final iterate.",
      iter
    ), call. = FALSE)
  }

  # Canonical signs make the reported basis reproducible without changing its
  # subspace, scores, objective, or convergence result.
  signs <- vapply(seq_len(m), function(j) {
    sign(W[which.max(abs(W[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  W <- sweep(W, 2L, signs, FUN = "*")
  current <- .whm_evaluate(W, statistics, denominator_tol, need_matrix = TRUE)

  MW <- current$matrix %*% W
  tangent_residual <- MW - W %*% crossprod(W, MW)
  stationarity_residual <- norm(tangent_residual, type = "F") /
    (1 + norm(MW, type = "F"))

  component_names <- paste0("WHMTR", seq_len(m))
  rownames(W) <- colnames(X)
  colnames(W) <- component_names
  scores <- X %*% W
  colnames(scores) <- component_names
  centered_scores <- sweep(scores, 2L, colMeans(scores), FUN = "-")
  score_sdev <- sqrt(colSums(centered_scores^2) / (nrow(X) - 1L))
  names(score_sdev) <- component_names
  preproc <- multivarious::fit(multivarious::pass(), X)

  multivarious::discriminant_projector(
    v = W,
    s = scores,
    sdev = score_sdev,
    preproc = preproc,
    labels = labels,
    classes = "whm_lda",
    objective = current$objective,
    objective_history = objective_history,
    iterations = iter,
    converged = converged,
    projector_change = projector_change,
    stationarity_residual = stationarity_residual,
    class_counts = statistics$class_counts,
    pair_classes = statistics$pair_classes,
    pair_weights = statistics$pair_weights,
    pair_denominators = current$denominators,
    pair_objectives = current$pair_objectives,
    m = m,
    max_iter = max_iter,
    tol = tol,
    denominator_tol = denominator_tol,
    seed = seed
  )
}


#' Cache the fixed class-pair quantities in the WHM-TR objective
#' @keywords internal
#' @noRd
.whm_prepare_statistics <- function(X, labels, class_levels) {
  groups <- lapply(class_levels, function(class_label) {
    X[labels == class_label, , drop = FALSE]
  })
  class_counts <- vapply(groups, nrow, integer(1L))
  class_means <- lapply(groups, colMeans)
  class_scatters <- lapply(seq_along(groups), function(j) {
    centered <- sweep(groups[[j]], 2L, class_means[[j]], FUN = "-")
    scatter <- crossprod(centered)
    if (any(!is.finite(scatter))) {
      stop("Within-class scatter overflowed; rescale 'X' before fitting.")
    }
    scatter
  })

  pair_indices <- utils::combn(length(class_levels), 2L)
  n_pairs <- ncol(pair_indices)
  pair_deltas <- vector("list", n_pairs)
  between_coefficients <- numeric(n_pairs)
  between_scales <- numeric(n_pairs)
  pair_weights <- numeric(n_pairs)

  for (pair in seq_len(n_pairs)) {
    j <- pair_indices[1L, pair]
    k <- pair_indices[2L, pair]
    delta <- class_means[[j]] - class_means[[k]]
    coefficient <- as.numeric(class_counts[j]) * class_counts[k] /
      (class_counts[j] + class_counts[k])
    between_scale <- coefficient * sum(delta^2)
    if (!is.finite(between_scale)) {
      stop(sprintf(
        "Between-class scatter overflowed for classes '%s' and '%s'; rescale 'X'.",
        class_levels[j], class_levels[k]
      ))
    }
    if (between_scale <= 0) {
      stop(sprintf(
        "Classes '%s' and '%s' have identical means, so their WHM-TR denominator is zero.",
        class_levels[j], class_levels[k]
      ))
    }

    pair_deltas[[pair]] <- delta
    between_coefficients[pair] <- coefficient
    between_scales[pair] <- between_scale
    pair_weights[pair] <- class_counts[j] + class_counts[k]
  }

  names(class_counts) <- class_levels
  pair_classes <- cbind(
    class_levels[pair_indices[1L, ]],
    class_levels[pair_indices[2L, ]]
  )
  colnames(pair_classes) <- c("class_j", "class_k")
  pair_names <- paste(pair_classes[, 1L], pair_classes[, 2L], sep = ":")
  rownames(pair_classes) <- pair_names
  names(pair_weights) <- pair_names
  names(between_coefficients) <- pair_names
  names(between_scales) <- pair_names

  list(
    class_counts = class_counts,
    class_scatters = class_scatters,
    pair_indices = pair_indices,
    pair_classes = pair_classes,
    pair_deltas = pair_deltas,
    pair_weights = pair_weights,
    between_coefficients = between_coefficients,
    between_scales = between_scales
  )
}


#' Initialize an orthonormal WHM-TR basis
#' @keywords internal
#' @noRd
.whm_initialize <- function(d, m, initial_W, seed) {
  if (!is.null(initial_W)) {
    initial_W <- as.matrix(initial_W)
    if (!is.numeric(initial_W) ||
        !identical(dim(initial_W), c(d, m)) ||
        any(!is.finite(initial_W))) {
      stop(sprintf(
        "'initial_W' must be a finite numeric matrix with dimensions %d by %d.",
        d, m
      ))
    }
    decomposition <- qr(initial_W, tol = sqrt(.Machine$double.eps))
    if (decomposition$rank < m) {
      stop("'initial_W' must have full column rank.")
    }
    return(qr.Q(decomposition)[, seq_len(m), drop = FALSE])
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (!is.null(seed)) {
    if (had_seed) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    }
    on.exit({
      if (had_seed) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed)
  }

  random_basis <- matrix(stats::rnorm(d * m), nrow = d, ncol = m)
  qr.Q(qr(random_basis))[, seq_len(m), drop = FALSE]
}


#' Evaluate the WHM-TR objective and Equation 31 matrix
#' @keywords internal
#' @noRd
.whm_evaluate <- function(W, statistics, denominator_tol, need_matrix = TRUE) {
  n_classes <- length(statistics$class_scatters)
  n_pairs <- ncol(statistics$pair_indices)
  class_traces <- vapply(statistics$class_scatters, function(scatter) {
    sum(W * (scatter %*% W))
  }, numeric(1L))

  denominators <- numeric(n_pairs)
  pair_objectives <- numeric(n_pairs)
  scatter_weights <- numeric(n_classes)
  if (need_matrix) {
    between_matrix <- matrix(0, nrow(W), nrow(W))
  }

  for (pair in seq_len(n_pairs)) {
    j <- statistics$pair_indices[1L, pair]
    k <- statistics$pair_indices[2L, pair]
    delta <- statistics$pair_deltas[[pair]]
    projected_delta <- drop(crossprod(delta, W))
    denominator <- statistics$between_coefficients[pair] *
      sum(projected_delta^2)
    minimum_denominator <- denominator_tol *
      statistics$between_scales[pair]
    if (!is.finite(denominator) || denominator <= minimum_denominator) {
      classes <- statistics$pair_classes[pair, ]
      stop(sprintf(
        paste0(
          "The projected WHM-TR denominator for classes '%s' and '%s' ",
          "is numerically zero; use another initialization or a larger subspace."
        ),
        classes[1L], classes[2L]
      ))
    }

    numerator <- statistics$pair_weights[pair] *
      (class_traces[j] + class_traces[k])
    pair_objectives[pair] <- numerator / denominator
    denominators[pair] <- denominator

    if (need_matrix) {
      within_scale <- statistics$pair_weights[pair] / denominator
      scatter_weights[j] <- scatter_weights[j] + within_scale
      scatter_weights[k] <- scatter_weights[k] + within_scale
      between_scale <- numerator * statistics$between_coefficients[pair] /
        denominator^2
      between_matrix <- between_matrix - between_scale * tcrossprod(delta)
    }
  }

  names(denominators) <- rownames(statistics$pair_classes)
  names(pair_objectives) <- rownames(statistics$pair_classes)
  result <- list(
    objective = sum(pair_objectives),
    denominators = denominators,
    pair_objectives = pair_objectives
  )
  if (need_matrix) {
    for (j in seq_len(n_classes)) {
      between_matrix <- between_matrix +
        scatter_weights[j] * statistics$class_scatters[[j]]
    }
    result$matrix <- 0.5 * (between_matrix + t(between_matrix))
  }
  result
}


#' Return the eigenvectors for the smallest eigenvalues of a symmetric matrix
#' @keywords internal
#' @noRd
.whm_bottom_eigen <- function(x, m) {
  d <- nrow(x)
  if (d <= 100L || m >= d - 1L) {
    decomposition <- eigen(x, symmetric = TRUE)
    take <- order(decomposition$values)[seq_len(m)]
    return(list(
      values = decomposition$values[take],
      vectors = decomposition$vectors[, take, drop = FALSE]
    ))
  }

  initial_vector <- sin(seq_len(d)) + cos(seq_len(d) * sqrt(2))
  initial_vector <- initial_vector / sqrt(sum(initial_vector^2))
  decomposition <- tryCatch(
    RSpectra::eigs_sym(
      x,
      k = m,
      which = "SA",
      opts = list(initvec = initial_vector, tol = 1e-10)
    ),
    error = function(e) NULL
  )
  if (is.null(decomposition) ||
      length(decomposition$values) != m ||
      any(!is.finite(decomposition$values)) ||
      any(!is.finite(decomposition$vectors))) {
    decomposition <- eigen(x, symmetric = TRUE)
    take <- order(decomposition$values)[seq_len(m)]
    return(list(
      values = decomposition$values[take],
      vectors = decomposition$vectors[, take, drop = FALSE]
    ))
  }

  take <- order(decomposition$values)
  list(
    values = decomposition$values[take],
    vectors = decomposition$vectors[, take, drop = FALSE]
  )
}
