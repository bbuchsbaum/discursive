test_that("geneig satisfies generalized eigen residuals and B normalization", {
  A <- matrix(c(6, 2, 1, 2, 5, 0.5, 1, 0.5, 3), 3, 3)
  B <- matrix(c(3, 0.4, 0.2, 0.4, 2, 0.1, 0.2, 0.1, 1), 3, 3)
  fit <- geneig(A, B, ncomp = 3)

  expect_equal(unname(crossprod(fit$v, B %*% fit$v)), diag(3), tolerance = 1e-10)
  for (j in 1:3) {
    residual <- A %*% fit$v[, j] - fit$values[j] * B %*% fit$v[, j]
    expect_lt(sqrt(sum(residual^2)), 1e-9)
  }
  expect_true(all(diff(fit$values) <= 0))
})

test_that("geneig robust path handles a singular positive-semidefinite B", {
  A <- diag(c(7, 3, 0))
  B <- diag(c(2, 1, 0))
  fit <- geneig(A, B, ncomp = 2)

  expect_equal(fit$rank_B, 2L)
  expect_equal(unname(fit$values), c(3.5, 3), tolerance = 1e-12)
  expect_equal(unname(crossprod(fit$v, B %*% fit$v)), diag(2), tolerance = 1e-12)
  expect_error(geneig(A, B, ncomp = 3), "ncomp")
  expect_error(geneig(A, B, ncomp = 2, method = "lapack"), "positive-definite")
})

test_that("geneig sdiag alias agrees with robust whitening", {
  A <- matrix(c(4, 1, 1, 2), 2)
  B <- diag(c(1, 3))
  robust <- geneig(A, B, ncomp = 2, method = "robust")
  sdiag <- geneig(A, B, ncomp = 2, method = "sdiag")

  expect_equal(sdiag$values, robust$values, tolerance = 1e-12)
  expect_equal(sdiag$v, robust$v, tolerance = 1e-12)
})

test_that("geneig rejects invalid matrix contracts", {
  expect_error(geneig(matrix(1:4, 2), diag(2), 1), "symmetric")
  expect_error(geneig(diag(2), diag(c(1, -1)), 1), "positive semidefinite")
  expect_error(geneig(diag(2), matrix(0, 2, 2), 1), "positive numerical rank")
  expect_error(geneig(diag(2), diag(3), 1), "same dimensions")
  expect_error(geneig(diag(2), diag(2), 1.5), "integer")
})
