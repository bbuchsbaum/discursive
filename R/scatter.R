.validate_scatter_matrix <- function(X) {
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("'X' must be a numeric matrix.", call. = FALSE)
  }
  if (nrow(X) < 1L || ncol(X) < 1L) {
    stop("'X' must have at least one row and one column.", call. = FALSE)
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.", call. = FALSE)
  }
  X
}

.validate_scatter_labels <- function(labels, n, min_classes = 1L) {
  if (length(labels) != n) {
    stop("'labels' must have length nrow(X).", call. = FALSE)
  }
  if ((!is.atomic(labels) && !is.factor(labels)) || !is.null(dim(labels))) {
    stop("'labels' must be an atomic vector or factor.", call. = FALSE)
  }
  if (anyNA(labels)) {
    stop("'labels' must not contain missing values.", call. = FALSE)
  }

  values <- as.character(labels)
  classes <- unique(values)
  if (length(classes) < min_classes) {
    stop(
      sprintf("'labels' must contain at least %s.",
              if (min_classes == 2L) "two classes" else "one class"),
      call. = FALSE
    )
  }

  codes <- match(values, classes)
  list(
    values = values,
    classes = classes,
    codes = codes,
    counts = tabulate(codes, nbins = length(classes))
  )
}

.scatter_result <- function(S, X = NULL, context) {
  if (any(!is.finite(S))) {
    stop(
      sprintf("The %s scatter overflowed; rescale 'X' before fitting.", context),
      call. = FALSE
    )
  }
  feature_names <- if (is.null(X)) NULL else colnames(X)
  if (!is.null(feature_names)) {
    dimnames(S) <- list(feature_names, feature_names)
  }
  S
}

.scatter_group_statistics <- function(X, labels, min_classes = 1L,
                                      compute_within = FALSE,
                                      compute_between = FALSE,
                                      keep_class_scatter = FALSE) {
  label_info <- .validate_scatter_labels(labels, nrow(X), min_classes)
  shifted_X <- sweep(X, 2L, X[1L, ], "-")
  if (any(!is.finite(shifted_X))) {
    stop("Pairwise differences in 'X' overflowed; rescale 'X'.", call. = FALSE)
  }
  group_sums <- rowsum(shifted_X, label_info$codes, reorder = TRUE)
  group_means <- sweep(group_sums, 1L, label_info$counts, "/")
  result <- c(
    label_info,
    list(shifted_means = group_means)
  )

  if (compute_within || keep_class_scatter) {
    centered <- shifted_X - group_means[label_info$codes, , drop = FALSE]
    if (compute_within) {
      result$within <- .scatter_result(crossprod(centered), X, "within-class")
    }
    if (keep_class_scatter) {
      result$class_scatter <- lapply(seq_along(label_info$classes), function(class_index) {
        rows <- label_info$codes == class_index
        .scatter_result(crossprod(centered[rows, , drop = FALSE]), X, "within-class")
      })
    }
  }

  if (compute_between) {
    overall_mean <- colMeans(shifted_X)
    offsets <- sweep(group_means, 2L, overall_mean, "-")
    weighted_offsets <- sweep(offsets, 1L, sqrt(label_info$counts), "*")
    result$between <- .scatter_result(
      crossprod(weighted_offsets),
      X,
      "between-class"
    )
  }
  result
}

.normalize_scatter_indices <- function(index, n, name) {
  if (is.logical(index)) {
    if (length(index) != n || anyNA(index)) {
      stop(sprintf("'%s' must be a non-missing logical vector of length nrow(X).", name),
           call. = FALSE)
    }
    index <- which(index)
  } else {
    if (!is.numeric(index) || anyNA(index) || any(!is.finite(index)) ||
        any(index != floor(index)) || any(index < 1) || any(index > n)) {
      stop(sprintf("'%s' must contain valid positive row indices.", name), call. = FALSE)
    }
    index <- as.integer(index)
  }

  if (length(index) == 0L) {
    stop(sprintf("'%s' must be non-empty.", name), call. = FALSE)
  }
  if (anyDuplicated(index)) {
    stop(sprintf("'%s' must not contain duplicate indices.", name), call. = FALSE)
  }
  index
}

#' Compute mean local scatter over sliding feature windows
#'
#' Centers each feature once, slides an odd-width window across the columns of
#' `X`, and averages the scatter matrices at corresponding positions within the
#' windows. For a window width `w`, the result is `w` by `w`; it is not a full
#' feature-by-feature scatter matrix.
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param window A finite odd integer between 3 and `ncol(X)`, inclusive.
#' @return A numeric `window` by `window` mean local scatter matrix.
#' @keywords internal
moving_scatter <- function(X, window) {
  X <- .validate_scatter_matrix(X)
  if (!is.numeric(window) || length(window) != 1L || is.na(window) ||
      !is.finite(window) || window != floor(window)) {
    stop("'window' must be a finite integer.", call. = FALSE)
  }
  if (window < 3L || window %% 2L == 0L) {
    stop("'window' must be an odd integer greater than or equal to 3.", call. = FALSE)
  }
  if (window > ncol(X)) {
    stop("'window' must not exceed the number of columns in 'X'.", call. = FALSE)
  }

  window <- as.integer(window)
  windows_count <- ncol(X) - window + 1L
  shifted_X <- sweep(X, 2L, X[1L, ], "-")
  if (any(!is.finite(shifted_X))) {
    stop("Pairwise differences in 'X' overflowed; rescale 'X'.", call. = FALSE)
  }
  centered <- sweep(shifted_X, 2L, colMeans(shifted_X), "-")
  scatter_sum <- matrix(0, nrow = window, ncol = window)

  for (start in seq_len(windows_count)) {
    columns <- start + seq_len(window) - 1L
    scatter_sum <- scatter_sum + crossprod(centered[, columns, drop = FALSE])
  }

  .scatter_result(scatter_sum / windows_count, context = "moving")
}

#' Calculate pooled within-class scatter
#'
#' Computes the unnormalized within-class scatter
#' \deqn{S_w = \sum_k \sum_{i \in k}(x_i - \bar{x}_k)(x_i - \bar{x}_k)^T.}
#' Class centering is performed explicitly before an optimized cross-product,
#' avoiding cancellation from subtracting two large raw-moment matrices.
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param Y An atomic vector or factor of class labels with length `nrow(X)`.
#' @return A numeric feature-by-feature within-class scatter matrix.
#' @keywords internal
pooled_scatter <- function(X, Y) {
  X <- .validate_scatter_matrix(X)
  .scatter_group_statistics(X, Y, compute_within = TRUE)$within
}

#' Calculate total scatter
#'
#' Computes the unnormalized total scatter
#' \deqn{S_t = \sum_i (x_i - \mu)(x_i - \mu)^T.}
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param mu A finite numeric vector of length `ncol(X)`.
#' @return A numeric feature-by-feature total scatter matrix.
#' @keywords internal
total_scatter <- function(X, mu) {
  X <- .validate_scatter_matrix(X)
  if (!is.numeric(mu) || length(mu) != ncol(X)) {
    stop("'mu' must be a numeric vector with length ncol(X).", call. = FALSE)
  }
  if (any(!is.finite(mu))) {
    stop("'mu' must contain only finite values.", call. = FALSE)
  }

  centered <- sweep(X, 2L, mu, "-")
  .scatter_result(crossprod(centered), X, "total")
}

#' Calculate between-class scatter
#'
#' Computes the unnormalized between-class scatter from class counts and means:
#' \deqn{S_b = \sum_k n_k (\bar{x}_k - \bar{x})(\bar{x}_k - \bar{x})^T.}
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param Y An atomic vector or factor of class labels with length `nrow(X)`.
#' @return A numeric feature-by-feature between-class scatter matrix.
#' @keywords internal
between_class_scatter <- function(X, Y) {
  X <- .validate_scatter_matrix(X)
  .scatter_group_statistics(X, Y, compute_between = TRUE)$between
}

#' Calculate within-class scatter
#'
#' Computes the unnormalized within-class scatter after centering observations
#' by their class means.
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param Y An atomic vector or factor of class labels with length `nrow(X)`.
#' @return A numeric feature-by-feature within-class scatter matrix.
#' @keywords internal
within_class_scatter <- function(X, Y) {
  X <- .validate_scatter_matrix(X)
  .scatter_group_statistics(X, Y, compute_within = TRUE)$within
}

#' Calculate binary between-class scatter
#'
#' For two disjoint, non-empty groups, computes the equivalent rank-one form
#' \deqn{S_b = \frac{n_1 n_2}{n_1 + n_2}
#' (\bar{x}_1 - \bar{x}_2)(\bar{x}_1 - \bar{x}_2)^T.}
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param id1 Positive integer indices or a logical vector selecting group one.
#' @param id2 Positive integer indices or a logical vector selecting group two.
#' @return A numeric feature-by-feature between-class scatter matrix.
#' @keywords internal
binary_between_scatter <- function(X, id1, id2) {
  X <- .validate_scatter_matrix(X)
  id1 <- .normalize_scatter_indices(id1, nrow(X), "id1")
  id2 <- .normalize_scatter_indices(id2, nrow(X), "id2")
  if (length(intersect(id1, id2)) > 0L) {
    stop("'id1' and 'id2' must be disjoint.", call. = FALSE)
  }

  n1 <- length(id1)
  n2 <- length(id2)
  shifted_X <- sweep(X, 2L, X[id1[1L], ], "-")
  if (any(!is.finite(shifted_X))) {
    stop("Pairwise differences in 'X' overflowed; rescale 'X'.", call. = FALSE)
  }
  mean_difference <- colMeans(shifted_X[id1, , drop = FALSE]) -
    colMeans(shifted_X[id2, , drop = FALSE])
  weight <- as.double(n1) * n2 / (n1 + n2)
  .scatter_result(weight * tcrossprod(mean_difference), X, "between-class")
}

#' Calculate scatter matrices for every pair of classes
#'
#' Computes within-class and between-class scatter for each unordered class
#' pair. Class means and within-class scatters are cached once, so observations
#' are not rescanned for every pair.
#'
#' @param X A numeric matrix with samples in rows and features in columns.
#' @param labels An atomic vector or factor of class labels with length `nrow(X)`.
#' @return A list with one element per unordered class pair. Each element
#'   contains `w`, the pooled within-class scatter, and `b`, the binary
#'   between-class scatter.
#' @keywords internal
pairwise_within_scatter <- function(X, labels) {
  X <- .validate_scatter_matrix(X)
  statistics <- .scatter_group_statistics(
    X,
    labels,
    min_classes = 2L,
    keep_class_scatter = TRUE
  )
  pairs <- utils::combn(seq_along(statistics$classes), 2L)

  lapply(seq_len(ncol(pairs)), function(i) {
    class1 <- pairs[1L, i]
    class2 <- pairs[2L, i]
    n1 <- statistics$counts[class1]
    n2 <- statistics$counts[class2]
    mean_difference <- statistics$shifted_means[class1, ] -
      statistics$shifted_means[class2, ]

    list(
      w = statistics$class_scatter[[class1]] + statistics$class_scatter[[class2]],
      b = .scatter_result(
        as.double(n1) * n2 / (n1 + n2) * tcrossprod(mean_difference),
        X,
        "between-class"
      )
    )
  })
}
