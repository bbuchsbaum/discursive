trace_ratio_diagonal_oracle <- function(a, b, p) {
  subsets <- combn(seq_along(a), p, simplify = FALSE)
  ratios <- vapply(subsets, function(index) sum(a[index]) / sum(b[index]), numeric(1L))
  best <- which.max(ratios)
  list(index = subsets[[best]], value = ratios[best])
}

test_that("trace_ratio matches a combinatorial ratio-of-traces oracle", {
  a <- c(10, 9, 1)
  b <- c(1, 3, 1)
  oracle <- trace_ratio_diagonal_oracle(a, b, p = 2)
  fit <- trace_ratio(diag(a), diag(b), ncomp = 2, eps = 1e-12)
  selected_mass <- rowSums(fit$v^2)

  expect_equal(fit$value, oracle$value, tolerance = 1e-12)
  expect_equal(order(selected_mass, decreasing = TRUE)[1:2], oracle$index)
  expect_equal(unname(crossprod(fit$v)), diag(2), tolerance = 1e-12)
  # The old sum-of-componentwise-ratios objective prefers coordinates 1 and 2.
  expect_false(setequal(oracle$index, order(a / b, decreasing = TRUE)[1:2]))
})

test_that("trace_ratio exposes coherent convergence and projection metadata", {
  A <- matrix(c(5, 1, 0, 1, 4, 0.5, 0, 0.5, 2), 3, 3)
  B <- matrix(c(2, 0.2, 0, 0.2, 1.5, 0.1, 0, 0.1, 1), 3, 3)
  X <- matrix(seq_len(24), ncol = 3)
  y <- rep(c("a", "b"), each = 4)
  fit <- trace_ratio(A, B, X = X, y = y, ncomp = 2, eps = 1e-10)

  direct_value <- sum(fit$v * (A %*% fit$v)) /
    sum(fit$v * (B %*% fit$v))
  expect_equal(fit$value, direct_value, tolerance = 1e-12)
  expect_equal(tail(fit$history, 1), fit$value, tolerance = 1e-12)
  expect_equal(length(fit$history), fit$iterations)
  expect_true(all(diff(fit$history) >= -1e-10))
  expect_equal(multivarious::project(fit, X), fit$s, tolerance = 1e-12)
})

test_that("trace_ratio leaves the caller RNG state unchanged", {
  set.seed(733)
  before <- .Random.seed
  fit1 <- trace_ratio(diag(c(8, 4, 1)), diag(c(1, 2, 3)), ncomp = 1)
  after <- .Random.seed
  fit2 <- trace_ratio(diag(c(8, 4, 1)), diag(c(1, 2, 3)), ncomp = 1)

  expect_identical(after, before)
  expect_identical(.Random.seed, before)
  expect_equal(fit1$v, fit2$v, tolerance = 0)
})

test_that("trace_ratio validates symmetry, denominator, and training dimensions", {
  A <- diag(3)
  B <- diag(3)
  nonsymmetric <- A
  nonsymmetric[1, 2] <- 1
  expect_error(trace_ratio(nonsymmetric, B), "symmetric")
  expect_error(trace_ratio(A, diag(c(1, -1, 1))), "positive semidefinite")
  expect_error(trace_ratio(A, matrix(0, 3, 3)), "positive numerical rank")
  expect_error(trace_ratio(A, B, ncomp = 1.5), "integer")
  expect_error(trace_ratio(A, B, X = matrix(1, 4, 2)), "nrow\\(A\\) columns")
  expect_error(trace_ratio(A, B, X = matrix(1, 4, 3), y = 1:3), "one non-missing")
})
