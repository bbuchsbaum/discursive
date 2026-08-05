#' Weighted Multi-label Linear Discriminant Analysis
#'
#' Fits the weighted multi-label LDA (wMLDA) framework of Xu (2018). A label
#' weight matrix is estimated using one of five strategies, then used to form
#' weighted within- and between-label scatter matrices. The generalized
#' eigenproblem is solved by symmetric whitening.
#'
#' @param X Numeric matrix or data frame with samples in rows and features in
#'   columns.
#' @param Y Binary numeric or logical matrix with one row per sample and one
#'   column per label. Every sample must have at least one label and every label
#'   must occur at least once.
#' @param weight_method Weighting strategy. One of `"binary"`, `"correlation"`,
#'   `"entropy"`, `"fuzzy"`, or `"dependence"`.
#' @param ncomp Optional positive integer giving the number of discriminant
#'   components. The default uses the positive numerical between-label rank,
#'   bounded by `ncol(Y) - 1`.
#' @param max_iter_fuzzy Maximum number of fuzzy-weight updates.
#' @param tol_fuzzy Positive convergence tolerance for fuzzy weights.
#' @param max_iter_dep Maximum number of randomized coordinate-descent epochs
#'   for dependence weights.
#' @param preproc A preprocessing specification from \pkg{multivarious}. The
#'   default centers the predictors.
#' @param reg Nonnegative relative ridge coefficient for the within-label
#'   scatter. The actual ridge is `reg` times a data-dependent scatter scale.
#' @param seed Optional nonnegative integer seed used by dependence weighting.
#'   The caller's random-number-generator state is restored after fitting.
#' @param tol Relative numerical-rank tolerance in `(0, 1)` for the final
#'   between-label eigenspectrum.
#'
#' @return A [multivarious::discriminant_projector()] with class `"wMLDA"`.
#'   In addition to the standard projector fields, it contains:
#'   \describe{
#'     \item{eigenvalues}{Generalized eigenvalues of the returned directions.}
#'     \item{weights}{The fitted sample-by-label weight matrix.}
#'     \item{label_matrix}{The complete binary training label matrix.}
#'     \item{weight_method}{The selected weighting strategy.}
#'     \item{ncomp, active_labels}{The fitted dimension and labels with
#'       positive fitted total weight.}
#'     \item{class_weights, instance_weights, total_weight}{Weight totals used
#'       to construct the scatters.}
#'     \item{between_rank, denominator_rank}{Numerical ranks diagnosed by the
#'       generalized eigensolver.}
#'     \item{ridge, reg, tol}{The fitted ridge and numerical controls.}
#'     \item{weight_iterations, weight_converged}{Weight-optimization
#'       diagnostics.}
#'   }
#'
#' @details
#' If `W` is the sample-by-label weight matrix, let `m` contain its row sums,
#' `n_k` its column sums, and `A = W diag(n_k^(-1/2))`. With preprocessed data
#' `X`, wMLDA uses
#' \deqn{S_w = X^T [diag(m) - A A^T] X / n_w}
#' and
#' \deqn{S_b = X^T [A A^T - m m^T / n_w] X / n_w,}
#' where `n_w = sum(m)`. The implementation evaluates these expressions using
#' feature-by-label cross-products, without materializing dense sample-by-sample
#' scatter operators.
#'
#' The returned directions `V` solve
#' \deqn{S_b V = (S_w + r I) V Lambda,}
#' where `r` is the fitted relative ridge. They are normalized so that
#' `t(V) (S_w + r I) V` is the identity, up to numerical precision.
#'
#' `labels` in the returned projector is the label with the largest fitted
#' weight for each training row. It is only a single-label adapter required by
#' the projector interface; `label_matrix` retains the full multi-label target.
#'
#' Dependence weighting uses the centered linear-kernel HSIC objective and a
#' randomized, discrete coordinate update over each sample's relevant labels.
#' It requires an `n`-by-`n` kernel matrix and therefore uses quadratic memory.
#'
#' @references
#' Xu, J. (2018). A weighted linear discriminant analysis framework for
#' multi-label feature extraction. *Neurocomputing*, 275, 107--120.
#' \doi{10.1016/j.neucom.2017.05.008}
#'
#' @seealso [multivarious::discriminant_projector()]
#' @export
#'
#' @examples
#' X <- as.matrix(iris[, 1:4])
#' Y <- cbind(
#'   setosa = iris$Species == "setosa",
#'   versicolor = iris$Species == "versicolor",
#'   virginica = iris$Species == "virginica",
#'   wide_sepal = iris$Sepal.Width > median(iris$Sepal.Width)
#' )
#' fit <- wMLDA(X, Y, weight_method = "entropy", ncomp = 2)
#' multivarious::project(fit, X[1:5, , drop = FALSE])
wMLDA <- function(X,
                  Y,
                  weight_method = c(
                    "binary", "correlation", "entropy", "fuzzy", "dependence"
                  ),
                  ncomp = NULL,
                  max_iter_fuzzy = 100,
                  tol_fuzzy = 1e-6,
                  max_iter_dep = 100,
                  preproc = multivarious::center(),
                  reg = 1e-9,
                  seed = NULL,
                  tol = 1e-6) {
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }
  if (!(is.numeric(Y) || is.logical(Y)) || length(dim(Y)) != 2L ||
      nrow(Y) != nrow(X) || ncol(Y) < 2L) {
    stop("'Y' must be a binary matrix with one row per sample and at least two labels.")
  }
  if (anyNA(Y) || any(!(Y %in% c(0, 1)))) {
    stop("'Y' must contain only non-missing binary values (0 or 1).")
  }
  storage.mode(Y) <- "double"
  if (any(rowSums(Y) == 0)) {
    stop("Every row of 'Y' must contain at least one relevant label.")
  }
  if (any(colSums(Y) == 0)) {
    stop("Every column of 'Y' must contain at least one relevant sample.")
  }

  label_names <- colnames(Y)
  if (is.null(label_names)) {
    label_names <- paste0("Label", seq_len(ncol(Y)))
  } else if (anyDuplicated(label_names)) {
    stop("Column names of 'Y' must be unique.")
  }
  colnames(Y) <- label_names

  weight_method <- match.arg(weight_method)
  max_iter_fuzzy <- .check_whole_number(
    max_iter_fuzzy, "max_iter_fuzzy", lower = 1L
  )
  tol_fuzzy <- .check_finite_scalar(
    tol_fuzzy, "tol_fuzzy", lower = 0, lower_open = TRUE
  )
  max_iter_dep <- .check_whole_number(max_iter_dep, "max_iter_dep", lower = 1L)
  reg <- .check_finite_scalar(reg, "reg", lower = 0)
  tol <- .check_finite_scalar(
    tol, "tol", lower = 0, upper = 1,
    lower_open = TRUE, upper_open = TRUE
  )
  if (!is.null(seed)) {
    seed <- .check_whole_number(
      seed, "seed", lower = 0L, upper = .Machine$integer.max
    )
  }

  max_components <- min(ncol(X), ncol(Y) - 1L)
  if (!is.null(ncomp)) {
    ncomp <- .check_whole_number(
      ncomp, "ncomp", lower = 1L, upper = max_components
    )
  }

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }

  Wmat <- .with_wmlda_seed(seed, switch(
    weight_method,
    binary = weight_binary(Y),
    correlation = weight_correlation(Y),
    entropy = weight_entropy(Y),
    fuzzy = weight_fuzzy(Xp, Y, max_iter = max_iter_fuzzy, tol = tol_fuzzy),
    dependence = weight_dependence(Xp, Y, max_iter = max_iter_dep)
  ))
  colnames(Wmat) <- label_names
  weight_iterations <- attr(Wmat, "iterations", exact = TRUE)
  weight_converged <- attr(Wmat, "converged", exact = TRUE)
  weight_objective <- attr(Wmat, "objective", exact = TRUE)
  if (is.null(weight_iterations)) {
    weight_iterations <- 0L
  }
  if (is.null(weight_converged)) {
    weight_converged <- TRUE
  }

  instance_weights <- rowSums(Wmat)
  class_weights <- colSums(Wmat)
  total_weight <- sum(instance_weights)
  if (!is.finite(total_weight) || total_weight <= 0 ||
      any(!is.finite(Wmat)) || any(Wmat < 0)) {
    stop("The selected weighting strategy produced invalid label weights.")
  }
  active_labels <- class_weights > 0
  if (sum(active_labels) < 2L) {
    stop("The selected weighting strategy retained fewer than two weighted labels.")
  }

  # These cross-products are algebraically identical to multiplication by the
  # n-by-n wMLDA scatter operators, but avoid constructing those dense matrices.
  normalized_weights <- sweep(
    Wmat[, active_labels, drop = FALSE],
    2L,
    sqrt(class_weights[active_labels]),
    FUN = "/"
  )
  class_crossproduct <- crossprod(Xp, normalized_weights)
  global_crossproduct <- drop(crossprod(Xp, instance_weights))
  weighted_X <- sweep(Xp, 1L, sqrt(instance_weights), FUN = "*")

  shared_scatter <- tcrossprod(class_crossproduct)
  within_scatter <- (crossprod(weighted_X) - shared_scatter) / total_weight
  between_scatter <- (
    shared_scatter - tcrossprod(global_crossproduct) / total_weight
  ) / total_weight
  within_scatter <- 0.5 * (within_scatter + t(within_scatter))
  between_scatter <- 0.5 * (between_scatter + t(between_scatter))

  ridge_scale <- max(mean(diag(within_scatter)), mean(diag(between_scatter)))
  if (!is.finite(ridge_scale) || ridge_scale <= 0) {
    stop("The weighted label scatters have zero numerical scale.")
  }
  ridge <- reg * ridge_scale
  denominator <- within_scatter
  diag(denominator) <- diag(denominator) + ridge

  denominator_decomp <- eigen(denominator, symmetric = TRUE)
  denominator_max <- max(denominator_decomp$values)
  if (!is.finite(denominator_max) || denominator_max <= 0) {
    stop("The regularized within-label scatter has no positive directions.")
  }
  denominator_cutoff <- if (ridge > 0) {
    .Machine$double.eps * max(1L, ncol(X)) * denominator_max
  } else {
    tol * denominator_max
  }
  keep_denominator <- denominator_decomp$values > denominator_cutoff
  denominator_rank <- sum(keep_denominator)
  if (denominator_rank == 0L) {
    stop("The regularized within-label scatter has zero numerical rank.")
  }

  whitening <- sweep(
    denominator_decomp$vectors[, keep_denominator, drop = FALSE],
    2L,
    sqrt(denominator_decomp$values[keep_denominator]),
    FUN = "/"
  )
  whitened_between <- crossprod(
    whitening, between_scatter %*% whitening
  )
  whitened_between <- 0.5 * (whitened_between + t(whitened_between))
  between_decomp <- eigen(whitened_between, symmetric = TRUE)
  between_max <- max(between_decomp$values)
  if (!is.finite(between_max) || between_max <= 0) {
    stop("The weighted label means have no positive between-label scatter.")
  }
  between_cutoff <- tol * between_max
  between_rank <- sum(between_decomp$values > between_cutoff)
  if (between_rank == 0L) {
    stop("The weighted label means have zero positive numerical rank.")
  }

  if (is.null(ncomp)) {
    ncomp <- min(max_components, between_rank)
  } else if (ncomp > between_rank) {
    stop(sprintf(
      "'ncomp' (%d) exceeds the numerical between-label rank (%d).",
      ncomp, between_rank
    ))
  }

  vectors <- whitening %*%
    between_decomp$vectors[, seq_len(ncomp), drop = FALSE]
  signs <- vapply(seq_len(ncomp), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")

  component_names <- paste0("WMLDA", seq_len(ncomp))
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  centered_scores <- sweep(scores, 2L, colMeans(scores), FUN = "-")
  score_sdev <- sqrt(colSums(centered_scores^2) / (nrow(X) - 1L))
  names(score_sdev) <- component_names
  eigenvalues <- between_decomp$values[seq_len(ncomp)]
  names(eigenvalues) <- component_names

  primary_labels <- factor(
    label_names[max.col(Wmat, ties.method = "first")],
    levels = label_names
  )
  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = primary_labels,
    classes = "wMLDA",
    eigenvalues = eigenvalues,
    weights = Wmat,
    label_matrix = Y,
    weight_method = weight_method,
    ncomp = ncomp,
    active_labels = label_names[active_labels],
    class_weights = class_weights,
    instance_weights = instance_weights,
    total_weight = total_weight,
    between_rank = between_rank,
    denominator_rank = denominator_rank,
    ridge = ridge,
    reg = reg,
    tol = tol,
    weight_iterations = weight_iterations,
    weight_converged = weight_converged,
    weight_objective = weight_objective
  )
}


# Weighting methods ---------------------------------------------------------

#' @keywords internal
#' @noRd
weight_binary <- function(Y) {
  Y
}

#' @keywords internal
#' @noRd
weight_correlation <- function(Y) {
  gram <- crossprod(Y)
  norms <- sqrt(diag(gram))
  correlation <- gram / tcrossprod(norms)
  correlation[!is.finite(correlation)] <- 0
  weights <- sweep(Y %*% correlation, 1L, rowSums(Y), FUN = "/")
  pmax(weights, 0)
}

#' @keywords internal
#' @noRd
weight_entropy <- function(Y) {
  sweep(Y, 1L, rowSums(Y), FUN = "/")
}

#' @keywords internal
#' @noRd
weight_fuzzy <- function(X, Y, max_iter = 100, tol = 1e-6) {
  max_iter <- .check_whole_number(max_iter, "max_iter", lower = 1L)
  tol <- .check_finite_scalar(tol, "tol", lower = 0, lower_open = TRUE)
  weights <- weight_entropy(Y)

  update_means <- function(current_weights, previous_means = NULL) {
    squared_weights <- current_weights^2
    totals <- colSums(squared_weights)
    active <- totals > 0
    if (is.null(previous_means)) {
      updated_means <- matrix(0, ncol(Y), ncol(X))
    } else {
      updated_means <- previous_means
    }
    updated_means[active, ] <- sweep(
      crossprod(squared_weights[, active, drop = FALSE], X),
      1L,
      totals[active],
      FUN = "/"
    )
    updated_means
  }

  means <- update_means(weights)
  converged <- FALSE
  for (iteration in seq_len(max_iter)) {
    distance_squared <- outer(
      rowSums(X^2), rowSums(means^2), FUN = "+"
    ) - 2 * tcrossprod(X, means)
    distance_squared <- pmax(distance_squared, 0)
    updated <- matrix(0, nrow(Y), ncol(Y))

    for (i in seq_len(nrow(Y))) {
      relevant <- which(Y[i, ] == 1)
      distances <- distance_squared[i, relevant]
      distance_scale <- max(distances)
      zero_tolerance <- .Machine$double.eps * ncol(X) * distance_scale
      zero_distance <- which(distances <= zero_tolerance)
      if (length(zero_distance) > 0L) {
        selected <- relevant[zero_distance]
        updated[i, selected] <- 1 / length(selected)
      } else {
        inverse_distance <- min(distances) / distances
        updated[i, relevant] <- inverse_distance / sum(inverse_distance)
      }
    }

    change <- max(abs(updated - weights))
    weights <- updated
    means <- update_means(weights, means)
    if (change < tol) {
      converged <- TRUE
      break
    }
  }

  attr(weights, "iterations") <- iteration
  attr(weights, "converged") <- converged
  weights
}

#' @keywords internal
#' @noRd
weight_dependence <- function(X, Y, max_iter = 100) {
  max_iter <- .check_whole_number(max_iter, "max_iter", lower = 1L)
  centered_X <- sweep(X, 2L, colMeans(X), FUN = "-")
  theta <- tcrossprod(centered_X)
  theta <- 0.5 * (theta + t(theta))
  weights <- weight_entropy(Y)
  objective <- numeric(max_iter + 1L)
  objective[1L] <- sum(weights * (theta %*% weights))
  converged <- FALSE

  for (epoch in seq_len(max_iter)) {
    old_weights <- weights
    for (i in sample.int(nrow(Y))) {
      relevant <- which(Y[i, ] == 1)
      # The diagonal contribution is constant across one-hot candidates. Remove
      # the current row so each update exactly maximizes its coordinate block.
      scores <- drop(theta[i, ] %*% weights[, relevant, drop = FALSE]) -
        theta[i, i] * weights[i, relevant]
      selected <- relevant[which.max(scores)]
      weights[i, ] <- 0
      weights[i, selected] <- 1
    }
    objective[epoch + 1L] <- sum(weights * (theta %*% weights))
    if (identical(weights, old_weights)) {
      converged <- TRUE
      break
    }
  }

  attr(weights, "iterations") <- epoch
  attr(weights, "converged") <- converged
  attr(weights, "objective") <- objective[seq_len(epoch + 1L)]
  weights
}

#' Evaluate an expression under a local seed
#' @keywords internal
#' @noRd
.with_wmlda_seed <- function(seed, expr) {
  if (is.null(seed)) {
    return(force(expr))
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
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
  force(expr)
}
