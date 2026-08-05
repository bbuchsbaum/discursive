test_that("shrinkage_da matches the underlying sda discriminant scores", {
  X <- as.matrix(iris[, 1:4])
  y <- iris$Species
  fit <- shrinkage_da(
    X, y, lambda = 0.2, lambda.var = 0.3, lambda.freqs = 0.4
  )
  Xp <- multivarious::reprocess(fit, X)
  reference <- sda::sda(
    Xp, y, lambda = 0.2, lambda.var = 0.3, lambda.freqs = 0.4,
    diagonal = FALSE, verbose = FALSE
  )
  expected <- sweep(Xp %*% t(reference$beta), 2L, reference$alpha, FUN = "+")

  expect_s3_class(fit, "shrinkage_da")
  expect_equal(fit$v, t(reference$beta), tolerance = 1e-12)
  expect_equal(fit$s, expected, tolerance = 1e-12)
  expect_equal(multivarious::project(fit, X), expected, tolerance = 1e-12)
  expect_equal(fit$regularization, reference$regularization, tolerance = 1e-12)
})

test_that("shrinkage_da estimates omitted shrinkage parameters", {
  X <- as.matrix(iris[, 1:4])
  fit <- shrinkage_da(X, iris$Species)

  expect_true(all(is.finite(fit$regularization)))
  expect_true(all(fit$regularization >= 0 & fit$regularization <= 1))
  expect_equal(multivarious::project(fit, X), fit$s, tolerance = 1e-12)
})

test_that("shrinkage_da centering gives translation-invariant scores", {
  X <- as.matrix(iris[, 1:4])
  shift <- c(100, -30, 20, 5)
  fit <- shrinkage_da(X, iris$Species, lambda = 0.2)
  shifted <- shrinkage_da(sweep(X, 2L, shift, FUN = "+"), iris$Species, lambda = 0.2)

  expect_equal(fit$s, shifted$s, tolerance = 1e-9)
  expect_equal(fit$v, shifted$v, tolerance = 1e-9)
})

test_that("shrinkage_da validates inputs and shrinkage ranges", {
  X <- as.matrix(iris[, 1:4])
  y <- iris$Species
  expect_error(shrinkage_da(X, y[-1]), "one non-missing")
  expect_error(shrinkage_da(X, rep("a", nrow(X))), "at least two classes")
  expect_error(shrinkage_da(X, y, lambda = -0.1), "lambda")
  expect_error(shrinkage_da(X, y, lambda.var = 1.1), "lambda.var")
  X[1, 1] <- Inf
  expect_error(shrinkage_da(X, y), "finite")
})
