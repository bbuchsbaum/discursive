whm_fixture <- function() {
  set.seed(2718)
  sizes <- c(5L, 11L, 23L)
  means <- rbind(
    alpha = c(-2.0, 0.3, 0.0, 0.4, -0.2),
    beta = c(0.2, 1.8, -0.5, 0.0, 0.3),
    gamma = c(2.1, -0.4, 1.2, -0.3, 0.1)
  )
  X <- do.call(rbind, lapply(seq_along(sizes), function(k) {
    noise <- matrix(rnorm(sizes[k] * ncol(means), sd = 0.55),
      ncol = ncol(means)
    )
    sweep(noise, 2L, means[k, ], FUN = "+")
  }))
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  labels <- factor(rep(rownames(means), sizes), levels = rownames(means))

  set.seed(314)
  initial_W <- qr.Q(qr(matrix(rnorm(ncol(X) * 2L), ncol(X), 2L)))
  list(X = X, labels = labels, initial_W = initial_W, sizes = sizes)
}


whm_projection <- function(W) {
  Q <- qr.Q(qr(W))
  tcrossprod(Q)
}


# Literal dense implementation of Equations 6, 7, 25, and 31 in Li et al.
# It deliberately does not use the optimized class-scatter representation.
whm_dense_reference <- function(X, labels, W) {
  labels <- droplevels(as.factor(labels))
  class_levels <- levels(labels)
  pairs <- utils::combn(seq_along(class_levels), 2L)
  M <- matrix(0, ncol(X), ncol(X))
  pair_objectives <- numeric(ncol(pairs))
  denominators <- numeric(ncol(pairs))

  for (pair in seq_len(ncol(pairs))) {
    j <- pairs[1L, pair]
    k <- pairs[2L, pair]
    Xj <- X[labels == class_levels[j], , drop = FALSE]
    Xk <- X[labels == class_levels[k], , drop = FALSE]
    nj <- nrow(Xj)
    nk <- nrow(Xk)
    muj <- colMeans(Xj)
    muk <- colMeans(Xk)
    pair_mean <- (nj * muj + nk * muk) / (nj + nk)

    centered_j <- sweep(Xj, 2L, muj, FUN = "-")
    centered_k <- sweep(Xk, 2L, muk, FUN = "-")
    Sw <- crossprod(centered_j) + crossprod(centered_k)
    Sb <- nj * tcrossprod(muj - pair_mean) +
      nk * tcrossprod(muk - pair_mean)
    A <- (nj + nk) * Sw

    numerator <- sum(W * (A %*% W))
    denominator <- sum(W * (Sb %*% W))
    pair_objectives[pair] <- numerator / denominator
    denominators[pair] <- denominator
    M <- M + A / denominator - numerator * Sb / denominator^2
  }

  pair_names <- apply(
    rbind(class_levels[pairs[1L, ]], class_levels[pairs[2L, ]]),
    2L,
    paste,
    collapse = ":"
  )
  names(pair_objectives) <- pair_names
  names(denominators) <- pair_names
  list(
    objective = sum(pair_objectives),
    pair_objectives = pair_objectives,
    denominators = denominators,
    matrix = 0.5 * (M + t(M))
  )
}


test_that("WHM-TR cached equations match a literal imbalanced-class oracle", {
  dat <- whm_fixture()
  W0 <- dat$initial_W
  direct_initial <- whm_dense_reference(dat$X, dat$labels, W0)
  direct_eigen <- eigen(direct_initial$matrix, symmetric = TRUE)
  take <- order(direct_eigen$values)[seq_len(ncol(W0))]
  expected_W <- direct_eigen$vectors[, take, drop = FALSE]
  direct_updated <- whm_dense_reference(dat$X, dat$labels, expected_W)

  fit <- NULL
  expect_warning(
    fit <- whm_lda(
      dat$X,
      dat$labels,
      m = 2,
      max_iter = 1,
      tol = 1e-15,
      initial_W = W0
    ),
    "did not converge"
  )

  expect_false(fit$converged)
  expect_equal(fit$iterations, 1L)
  expect_equal(length(fit$objective_history), 2L)
  expect_equal(fit$objective_history[1L], direct_initial$objective,
    tolerance = 1e-10
  )
  expect_equal(fit$objective_history[2L], direct_updated$objective,
    tolerance = 1e-10
  )
  expect_equal(whm_projection(fit$v), whm_projection(expected_W),
    tolerance = 1e-9
  )
  expect_equal(unname(fit$pair_denominators),
    unname(direct_updated$denominators), tolerance = 1e-10
  )
  expect_equal(unname(fit$pair_objectives),
    unname(direct_updated$pair_objectives), tolerance = 1e-10
  )

  expected_weights <- c(
    alpha.beta = dat$sizes[1L] + dat$sizes[2L],
    alpha.gamma = dat$sizes[1L] + dat$sizes[3L],
    beta.gamma = dat$sizes[2L] + dat$sizes[3L]
  )
  expect_equal(unname(fit$pair_weights), unname(expected_weights))

  statistics <- discursive:::.whm_prepare_statistics(
    dat$X, dat$labels, levels(dat$labels)
  )
  optimized <- discursive:::.whm_evaluate(
    W0, statistics, denominator_tol = 1e-12, need_matrix = TRUE
  )
  expect_equal(optimized$matrix, direct_initial$matrix, tolerance = 1e-10)
  expect_equal(optimized$objective, direct_initial$objective,
    tolerance = 1e-10
  )
})


test_that("WHM-TR returns a converged orthonormal projector with diagnostics", {
  dat <- whm_fixture()
  fit <- whm_lda(
    dat$X, dat$labels, m = 2, initial_W = dat$initial_W,
    max_iter = 200, tol = 1e-8
  )

  expect_s3_class(fit, "whm_lda")
  expect_s3_class(fit, "discriminant_projector")
  expect_true(fit$converged)
  expect_lte(fit$iterations, fit$max_iter)
  expect_equal(length(fit$objective_history), fit$iterations + 1L)
  expect_true(all(is.finite(fit$objective_history)))
  expect_equal(fit$objective, tail(fit$objective_history, 1L),
    tolerance = 1e-10
  )
  expect_equal(fit$objective, sum(fit$pair_objectives), tolerance = 1e-12)
  expect_lte(fit$projector_change, fit$tol * sqrt(fit$m))
  expect_lt(fit$stationarity_residual, 1e-6)
  expect_equal(unname(crossprod(fit$v)), diag(2), tolerance = 1e-10)
  expect_true(all(is.finite(fit$v)))
  expect_true(all(is.finite(fit$s)))
  expect_equal(
    multivarious::project(fit, dat$X),
    fit$s,
    tolerance = 1e-12
  )
  expect_equal(fit$class_counts, setNames(dat$sizes, levels(dat$labels)))
  expect_equal(dim(fit$pair_classes), c(3L, 2L))
})


test_that("WHM-TR is invariant to translation, uniform scale, rows, and labels", {
  dat <- whm_fixture()
  control <- whm_lda(
    dat$X, dat$labels, 2, initial_W = dat$initial_W,
    max_iter = 200, tol = 1e-8
  )
  translation <- c(10, -7, 3, 0.5, 9)
  translated <- whm_lda(
    sweep(dat$X, 2L, translation, FUN = "+"),
    dat$labels,
    2,
    initial_W = dat$initial_W,
    max_iter = 200,
    tol = 1e-8
  )
  scaled <- whm_lda(
    dat$X * 7.5,
    dat$labels,
    2,
    initial_W = dat$initial_W,
    max_iter = 200,
    tol = 1e-8
  )
  permutation <- rev(seq_len(nrow(dat$X)))
  permuted <- whm_lda(
    dat$X[permutation, , drop = FALSE],
    dat$labels[permutation],
    2,
    initial_W = dat$initial_W,
    max_iter = 200,
    tol = 1e-8
  )
  relabeled <- whm_lda(
    dat$X,
    factor(c(alpha = "z", beta = "x", gamma = "y")[as.character(dat$labels)]),
    2,
    initial_W = dat$initial_W,
    max_iter = 200,
    tol = 1e-8
  )

  reference_projection <- whm_projection(control$v)
  expect_equal(whm_projection(translated$v), reference_projection,
    tolerance = 1e-7
  )
  expect_equal(whm_projection(scaled$v), reference_projection,
    tolerance = 1e-7
  )
  expect_equal(whm_projection(permuted$v), reference_projection,
    tolerance = 1e-7
  )
  expect_equal(whm_projection(relabeled$v), reference_projection,
    tolerance = 1e-7
  )
})


test_that("seeded WHM-TR initialization is reproducible and RNG-isolated", {
  dat <- whm_fixture()
  set.seed(808)
  state_before <- .Random.seed
  first <- whm_lda(dat$X, dat$labels, 2, seed = 29, tol = 1e-8)
  expect_identical(.Random.seed, state_before)
  second <- whm_lda(dat$X, dat$labels, 2, seed = 29, tol = 1e-8)
  expect_identical(.Random.seed, state_before)

  expect_equal(first$v, second$v, tolerance = 0)
  expect_equal(first$objective_history, second$objective_history, tolerance = 0)
  expect_error(
    whm_lda(
      dat$X, dat$labels, 2,
      initial_W = dat$initial_W,
      seed = 1
    ),
    "must be NULL"
  )
})


test_that("WHM-TR validates data, dimensions, labels, and solver controls", {
  dat <- whm_fixture()
  expect_error(whm_lda(letters[1:5], dat$labels, 2), "numeric matrix")

  nonfinite <- dat$X
  nonfinite[1L, 1L] <- Inf
  expect_error(whm_lda(nonfinite, dat$labels, 2), "finite values")
  nonfinite[1L, 1L] <- NA_real_
  expect_error(whm_lda(nonfinite, dat$labels, 2), "finite values")

  expect_error(whm_lda(dat$X, dat$labels[-1L], 2), "one value per row")
  missing_labels <- dat$labels
  missing_labels[1L] <- NA
  expect_error(whm_lda(dat$X, missing_labels, 2), "must not contain missing")
  expect_error(
    whm_lda(dat$X, rep("one", nrow(dat$X)), 2),
    "at least two observed"
  )

  expect_error(whm_lda(dat$X, dat$labels, 0), "must lie")
  expect_error(whm_lda(dat$X, dat$labels, 1.5), "must be an integer")
  expect_error(whm_lda(dat$X, dat$labels, ncol(dat$X)), "must lie")
  expect_error(whm_lda(dat$X, dat$labels, 2, max_iter = 0), "must lie")
  expect_error(whm_lda(dat$X, dat$labels, 2, tol = 0), "must lie")
  expect_error(
    whm_lda(dat$X, dat$labels, 2, denominator_tol = 1),
    "must lie"
  )

  expect_error(
    whm_lda(dat$X, dat$labels, 2, initial_W = diag(ncol(dat$X))),
    "dimensions"
  )
  rank_deficient <- cbind(rep(1, ncol(dat$X)), rep(1, ncol(dat$X)))
  expect_error(
    whm_lda(dat$X, dat$labels, 2, initial_W = rank_deficient),
    "full column rank"
  )

  labels_with_unused_level <- factor(
    dat$labels,
    levels = c(levels(dat$labels), "unused")
  )
  fit <- whm_lda(
    as.data.frame(dat$X), labels_with_unused_level, 2,
    initial_W = dat$initial_W
  )
  expect_identical(levels(fit$labels), levels(dat$labels))
})


test_that("WHM-TR reports zero and projected-zero pair denominators clearly", {
  equal_mean_X <- rbind(
    c(-1, 0), c(1, 0),
    c(0, -1), c(0, 1)
  )
  expect_error(
    whm_lda(equal_mean_X, factor(c("a", "a", "b", "b")), 1, seed = 1),
    "identical means"
  )

  separated_X <- rbind(
    c(-2, -1), c(-2, 1),
    c(2, -1), c(2, 1)
  )
  expect_error(
    whm_lda(
      separated_X,
      factor(c("a", "a", "b", "b")),
      1,
      initial_W = matrix(c(0, 1), 2, 1)
    ),
    "projected WHM-TR denominator"
  )
})


test_that("large-matrix eigensolver agrees with the full symmetric oracle", {
  values <- seq(-3, 7, length.out = 101L)
  x <- diag(values)
  partial <- discursive:::.whm_bottom_eigen(x, 3L)
  full <- eigen(x, symmetric = TRUE)
  expected <- sort(full$values)[seq_len(3L)]

  expect_equal(partial$values, expected, tolerance = 1e-9)
  expect_equal(
    sort(diag(crossprod(partial$vectors, x %*% partial$vectors))),
    expected,
    tolerance = 1e-8
  )
  expect_equal(crossprod(partial$vectors), diag(3), tolerance = 1e-8)
})
