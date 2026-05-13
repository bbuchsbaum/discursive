library(testthat)

test_that("print.discriminant_projector emits a compact summary with method name", {
  set.seed(1)
  n <- 30; d <- 4
  X <- matrix(rnorm(n * d), n, d)
  y <- factor(rep(1:3, length.out = n))
  fit <- ulda(X, y)

  expect_s3_class(fit, "discriminant_projector")
  expect_identical(fit$method, "ulda")
  expect_true(!is.null(fit$call))

  out <- capture.output(print(fit))
  expect_true(any(grepl("ulda", out, fixed = TRUE)))
  expect_true(any(grepl("Input dim",  out, fixed = TRUE)))
  expect_true(any(grepl("Output dim", out, fixed = TRUE)))
  expect_true(any(grepl("Samples",    out, fixed = TRUE)))
  expect_true(any(grepl("Classes",    out, fixed = TRUE)))
})

test_that("summary.discriminant_projector returns a structured summary", {
  set.seed(2)
  n <- 30; d <- 4
  X <- matrix(rnorm(n * d), n, d)
  y <- factor(rep(1:2, length.out = n))
  fit <- ulda(X, y)

  s <- summary(fit)
  expect_s3_class(s, "summary.discriminant_projector")
  expect_identical(s$method, "ulda")
  expect_equal(s$n, n)
  expect_equal(s$d_in, d)
  expect_equal(s$nclasses, 2L)
})
