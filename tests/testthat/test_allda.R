library(testthat)

set.seed(123)

test_that("allda runs on small numeric data", {
  n <- 20
  d <- 5
  X <- matrix(rnorm(n * d), nrow = n)
  y <- factor(rep(1:2, length.out = n))
  res <- allda(X, y, ncomp = 2, k = 3, max_iter = 2)
  expect_s3_class(res, "discriminant_projector")
  expect_equal(ncol(res$v), 2)
})

test_that("var_retained is validated", {
  n <- 10
  d <- 3
  X <- matrix(rnorm(n * d), nrow = n)
  y <- factor(rep(1:2, length.out = n))
  expect_error(allda(X, y, ncomp = 2, k = 2, var_retained = 1.5),
               "between 0 and 1")
})
