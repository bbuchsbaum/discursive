library(testthat)

test_that("dtrda runs on the iris data set and returns a projector", {
  data(iris)
  set.seed(42)
  X <- as.matrix(iris[, -5])
  y <- iris$Species

  res <- dtrda(X, y, ncomp = 2, alpha = 0.5)

  expect_s3_class(res, "discriminant_projector")
  expect_s3_class(res, "dtrda")
  expect_equal(ncol(res$v), 2L)
  expect_equal(nrow(res$s), nrow(X))
})

test_that("dtrda validates alpha and ncomp", {
  X <- matrix(rnorm(40), 20, 2)
  y <- factor(rep(1:2, 10))
  expect_error(dtrda(X, y, ncomp = 1, alpha = 1.5), "alpha")
  expect_error(dtrda(X, y, ncomp = 5, alpha = 0.5), "ncomp")
})
