slrr_fixture <- function() {
  set.seed(1401)
  y <- factor(c(rep("a", 3), rep("b", 5), rep("c", 8)))
  X <- rbind(
    matrix(rnorm(3 * 4, 0, 1), 3, 4),
    matrix(rnorm(5 * 4, 2, 1), 5, 4),
    matrix(rnorm(8 * 4, c(-1, 1, 2, -2), 1), 8, 4)
  )
  list(X = X, y = y)
}

literal_lrrr <- function(X, y, rank, lambda) {
  Xc <- scale(X, center = TRUE, scale = FALSE)
  G <- model.matrix(~ y - 1)
  G <- sweep(G, 2, sqrt(colSums(G)), "/")
  XtX <- crossprod(Xc)
  XtG <- crossprod(Xc, G)
  Sb <- tcrossprod(XtG)
  denominator <- XtX + lambda * diag(ncol(X))

  denominator_eig <- eigen(denominator, symmetric = TRUE)
  whitening <- sweep(
    denominator_eig$vectors,
    2,
    sqrt(denominator_eig$values),
    "/"
  )
  reduced <- crossprod(whitening, Sb %*% whitening)
  reduced_eig <- eigen(0.5 * (reduced + t(reduced)), symmetric = TRUE)
  A <- whitening %*% reduced_eig$vectors[, seq_len(rank), drop = FALSE]
  B <- solve(crossprod(A, denominator %*% A), crossprod(A, XtG))

  list(
    W = A %*% B,
    A = A,
    B = B,
    values = reduced_eig$values[seq_len(rank)],
    denominator = denominator,
    Sb = Sb,
    Xc = Xc,
    G = G
  )
}

test_that("LRRR matches the paper equations on unbalanced classes", {
  fixture <- slrr_fixture()
  reference <- literal_lrrr(fixture$X, fixture$y, rank = 2, lambda = 3)
  fit <- slrr(fixture$X, fixture$y, s = 2, lambda = 3, penalty = "ridge")

  expect_equal(unname(fit$v), unname(reference$W), tolerance = 1e-9)
  expect_equal(unname(fit$B), unname(reference$B), tolerance = 1e-9)
  expect_equal(fit$objective_history, {
    residual <- reference$G - reference$Xc %*% reference$W
    sum(residual^2) + 3 * sum(reference$W^2)
  }, tolerance = 1e-10)
})

test_that("LRRR basis satisfies the regularized generalized eigenproblem", {
  fixture <- slrr_fixture()
  fit <- slrr(fixture$X, fixture$y, s = 2, lambda = 0.4)
  reference <- literal_lrrr(fixture$X, fixture$y, rank = 2, lambda = 0.4)

  expect_equal(
    crossprod(fit$A, reference$denominator %*% fit$A),
    diag(2),
    tolerance = 1e-8
  )
  residual <- reference$Sb %*% fit$A -
    reference$denominator %*% fit$A %*% diag(fit$eigenvalues)
  expect_lt(max(abs(residual)), 1e-7)
})

test_that("SLRR objective is non-increasing and returns its learned factors", {
  fixture <- slrr_fixture()
  fit <- slrr(
    fixture$X, fixture$y, s = 2, lambda = 0.2,
    penalty = "l21", max_iter = 30, tol = 1e-10
  )

  expect_equal(dim(fit$A), c(4, 2))
  expect_equal(dim(fit$B), c(2, 3))
  expect_equal(fit$v, fit$A %*% fit$B, tolerance = 1e-10)
  expect_equal(length(fit$objective_history), fit$iterations)
  expect_true(all(is.finite(fit$objective_history)))
  expect_true(all(diff(fit$objective_history) <= 1e-9))
})

test_that("regularized SLRR variants work when features outnumber samples", {
  set.seed(87)
  X <- matrix(rnorm(12 * 30), 12, 30)
  y <- factor(rep(1:3, each = 4))

  for (penalty in c("ridge", "l21")) {
    fit <- slrr(
      X, y, s = 2, lambda = 0.1, penalty = penalty,
      max_iter = 8
    )
    expect_equal(dim(fit$v), c(30, 3), info = penalty)
    expect_true(all(is.finite(fit$v)), info = penalty)
    expect_true(qr(fit$v)$rank <= 2, info = penalty)
  }
})

test_that("randomized SLRR fits preserve rank, finiteness, and descent", {
  for (seed in 1:5) {
    set.seed(seed)
    X <- matrix(rnorm(24 * 5), 24, 5)
    y <- factor(rep(1:3, each = 8))
    X[y == 2, 1:2] <- X[y == 2, 1:2] + 0.5
    X[y == 3, 3:4] <- X[y == 3, 3:4] - 0.5
    fit <- slrr(
      X, y, s = 2, lambda = 0.15, penalty = "l21",
      max_iter = 15, tol = 1e-9
    )

    expect_true(all(is.finite(fit$v)), info = paste("seed", seed))
    expect_true(qr(fit$v)$rank <= 2, info = paste("seed", seed))
    expect_true(
      all(diff(fit$objective_history) <= 1e-8),
      info = paste("seed", seed)
    )
  }
})

test_that("SLRR preprocessing and projection are coherent and translation invariant", {
  fixture <- slrr_fixture()
  shift <- c(100, -20, 5, 80)
  shifted <- sweep(fixture$X, 2, shift, "+")

  fit <- slrr(fixture$X, fixture$y, s = 2, lambda = 0.3)
  fit_shifted <- slrr(shifted, fixture$y, s = 2, lambda = 0.3)

  expect_equal(multivarious::project(fit, fixture$X), fit$s, tolerance = 1e-10)
  expect_equal(fit_shifted$v, fit$v, tolerance = 1e-9)
  expect_equal(fit_shifted$s, fit$s, tolerance = 1e-8)
})

test_that("SLRR validates rank, penalties, iterations, and labels", {
  fixture <- slrr_fixture()
  X <- fixture$X
  y <- fixture$y

  expect_error(slrr(X, y, s = 0), "'s'")
  expect_error(slrr(X, y, s = 1.5), "integer")
  expect_error(slrr(X, y, s = 3), "'s'")
  expect_error(slrr(X, y, s = 2, lambda = -1), "lambda")
  expect_error(slrr(X, y, s = 2, max_iter = 0), "max_iter")
  expect_error(slrr(X, y, s = 2, tol = 0), "tol")
  expect_error(slrr(X, rep("a", nrow(X)), s = 1), "at least two classes")
  expect_error(slrr(X, y[-1], s = 2), "one non-missing label")
  X[1, 1] <- Inf
  expect_error(slrr(X, y, s = 2), "finite")
})
