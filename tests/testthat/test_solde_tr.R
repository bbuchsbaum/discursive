literal_solde_graphs <- function(X, y, s, sigma) {
  n <- nrow(X)
  distances <- as.matrix(dist(X))
  diag(distances) <- Inf
  S <- H <- F <- matrix(0, n, n)

  for (i in seq_len(n)) {
    neighbors <- order(distances[i, ])[seq_len(s)]
    for (j in neighbors) {
      kernel <- exp(-(distances[i, j]^2) / sigma)
      if (y[i] == y[j]) {
        S[i, j] <- S[j, i] <- kernel
        H[i, j] <- H[j, i] <- 1 - kernel
      } else {
        F[i, j] <- F[j, i] <- kernel
      }
    }
  }
  list(S = S, H = H, F = F)
}

test_that("SOLDE graph construction matches a literal nearest-neighbor oracle", {
  X <- rbind(
    c(0.0, 0.0), c(0.8, 0.1), c(2.1, 0.2),
    c(0.2, 2.0), c(1.4, 2.4), c(3.0, 2.2)
  )
  y <- factor(c("a", "a", "b", "a", "b", "b"))
  expected <- literal_solde_graphs(X, y, s = 2, sigma = 1.7)
  observed <- .solde_adjacency_graphs(X, y, s = 2, sigma = 1.7)

  expect_s4_class(observed$S, "sparseMatrix")
  expect_equal(as.matrix(observed$S), expected$S, tolerance = 1e-12)
  expect_equal(as.matrix(observed$H), expected$H, tolerance = 1e-12)
  expect_equal(as.matrix(observed$F), expected$F, tolerance = 1e-12)
})

test_that("SOLDE graph construction excludes self-neighbors under duplicate ties", {
  X <- rbind(
    c(0, 0), c(0, 0), c(0, 0),
    c(1, 0), c(2, 0), c(3, 0)
  )
  y <- factor(c(1, 1, 1, 1, 2, 2))
  graphs <- .solde_adjacency_graphs(X, y, s = 2, sigma = 1)

  for (graph in graphs) {
    expect_equal(diag(graph), rep(0, nrow(X)))
    expect_equal(graph, t(graph))
    expect_true(all(is.finite(graph@x)))
  }
})

test_that("trace-ratio solver matches the analytic one-component solution", {
  numerator <- diag(c(9, 4, 1))
  denominator <- diag(c(1, 2, 3))
  fit <- .solde_trace_ratio(
    numerator, denominator, ncomp = 1, tol = 1e-12, maxit = 30
  )

  expect_equal(abs(drop(fit$vectors)), c(1, 0, 0), tolerance = 1e-10)
  expect_equal(fit$value, 9, tolerance = 1e-10)
  expect_true(all(diff(fit$history) >= -1e-12))
  expect_equal(crossprod(fit$vectors), matrix(1), tolerance = 1e-12)
})

test_that("SOLDE-TR regularizes an indefinite denominator and improves its ratio", {
  set.seed(91)
  X <- matrix(rnorm(40 * 6), 40, 6)
  y <- factor(rep(1:4, each = 10))
  expect_warning(
    fit <- solde_tr(X, y, s = 4, m = 3, maxit = 50),
    "materially indefinite"
  )

  expect_gt(fit$denominator_shift, 0)
  expect_gt(fit$denominator_min_eigenvalue, 0)
  expect_true(all(diff(fit$trace_ratio_history) >= -1e-9))
  expect_equal(length(fit$trace_ratio_history), fit$iterations)
  expect_true(fit$converged)
})

test_that("randomized SOLDE-TR fits preserve finite orthogonal solutions", {
  for (seed in 1:5) {
    set.seed(seed + 100)
    X <- matrix(rnorm(36 * 5), 36, 5)
    y <- factor(rep(1:3, each = 12))
    X[y == 2, 1:2] <- X[y == 2, 1:2] + 0.7
    X[y == 3, 3:4] <- X[y == 3, 3:4] - 0.7
    fit <- suppressWarnings(solde_tr(X, y, s = 3, m = 2, maxit = 40))

    expect_true(all(is.finite(fit$v)), info = paste("seed", seed))
    expect_equal(
      crossprod(fit$W_solde_tr), diag(2), tolerance = 1e-8,
      info = paste("seed", seed)
    )
    expect_true(
      all(diff(fit$trace_ratio_history) >= -1e-8),
      info = paste("seed", seed)
    )
  }
})

test_that("SOLDE graph storage remains sparse at representative size", {
  set.seed(908)
  n <- 200
  X <- matrix(rnorm(n * 6), n, 6)
  y <- factor(rep(1:4, each = n / 4))
  graphs <- .solde_adjacency_graphs(X, y, s = 4, sigma = 2)

  for (graph in graphs) {
    expect_s4_class(graph, "sparseMatrix")
    expect_lt(as.numeric(object.size(graph)), 8 * n^2 / 4)
  }
})

test_that("SOLDE-TR returns its documented factors without changing RNG state", {
  X <- as.matrix(iris[, 1:4])
  y <- iris$Species
  set.seed(733)
  rng_before <- .Random.seed
  fit <- solde_tr(X, y, s = 3, m = 2)
  rng_after <- .Random.seed

  expect_identical(rng_after, rng_before)
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(dim(fit$W_pca), c(4, 4))
  expect_equal(dim(fit$W_solde_tr), c(4, 2))
  expect_length(fit$eigvals_pca, 4)
  expect_equal(fit$v, fit$W_pca %*% fit$W_solde_tr, tolerance = 1e-10)
  expect_equal(crossprod(fit$W_solde_tr), diag(2), tolerance = 1e-10)
  expect_equal(multivarious::project(fit, X), fit$s, tolerance = 1e-10)
})

test_that("SOLDE-TR is translation invariant after fitted centering", {
  set.seed(440)
  X <- matrix(rnorm(90 * 5), 90, 5)
  y <- factor(rep(1:3, each = 30))
  X[y == 2, 1:2] <- X[y == 2, 1:2] + 1
  X[y == 3, 3:4] <- X[y == 3, 3:4] - 1
  shifted <- sweep(X, 2, c(100, -50, 25, 10, -5), "+")
  fit <- suppressWarnings(solde_tr(X, y, s = 3, m = 2))
  fit_shifted <- suppressWarnings(solde_tr(shifted, y, s = 3, m = 2))

  expect_equal(
    tcrossprod(fit$v), tcrossprod(fit_shifted$v), tolerance = 1e-8
  )
  expect_equal(
    as.matrix(dist(fit$s)), as.matrix(dist(fit_shifted$s)), tolerance = 1e-8
  )
})

test_that("SOLDE-TR validates graph and optimization parameters", {
  X <- as.matrix(iris[1:30, 1:4])
  y <- factor(rep(c("a", "b"), each = 15))

  expect_error(solde_tr(X, y, s = 0, m = 1), "'s'")
  expect_error(solde_tr(X, y, s = 1.5, m = 1), "integer")
  expect_error(solde_tr(X, y, s = 2, m = 1, sigma = 0), "sigma")
  expect_error(solde_tr(X, y, s = 2, m = 1, alpha = 0.4), "alpha")
  expect_error(solde_tr(X, y, s = 2, m = 0), "'m'")
  expect_error(solde_tr(X, y, s = 2, m = 1, maxit = 0), "maxit")
  expect_error(solde_tr(X, y, s = 2, m = 1, tol = 0), "tol")
  expect_error(solde_tr(X, rep("a", nrow(X)), s = 2, m = 1), "at least two classes")
  expect_error(solde_tr(X, y[-1], s = 2, m = 1), "one non-missing label")
})
