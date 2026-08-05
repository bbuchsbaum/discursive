test_that("dtrda runs and returns a valid projector on iris", {
  data(iris)
  X <- as.matrix(iris[, -5])
  Y <- iris$Species

  fit <- dtrda(X, Y, d = 2, alpha = 0.5)

  expect_s3_class(fit, "dtrda")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(dim(fit$v), c(ncol(X), 2))
  expect_equal(dim(fit$s), c(nrow(X), 2))
  expect_equal(length(fit$sdev), 2)
})

test_that("dtrda validates target dimension", {
  data(iris)
  X <- as.matrix(iris[, -5])
  Y <- iris$Species

  expect_error(
    dtrda(X, Y, d = 4, alpha = 0.5),
    "d must be less than the number of classes minus 1"
  )
})
