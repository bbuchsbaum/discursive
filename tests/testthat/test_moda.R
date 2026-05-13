# tests/testthat/test-moda_full.R

library(testthat)

test_that("moda_full runs on a small synthetic dataset (random init, single cluster)", {
  set.seed(123)
  d <- 10
  n <- 20
  # n x d, the standardized discursive orientation
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y <- rep(c(1, 2), each = n / 2)

  result <- moda_full(X, y, ncomp = 2,
                      numClusters = 1,
                      pcaFirst = FALSE,
                      B_init = "random",
                      maxIter = 10,
                      verbose = FALSE)

  expect_s3_class(result, "moda_full")
  expect_s3_class(result, "discriminant_projector")

  expect_equal(dim(result$v), c(d, 2))
  expect_equal(dim(result$s), c(n, 2))

  expect_true(length(result$objVals) > 0)

  for (cl_vec in result$clusters) {
    expect_true(all(cl_vec == 1))
  }
})

test_that("moda_full runs with PCA initialization on synthetic data (multiple clusters)", {
  set.seed(123)
  d <- 12
  n <- 24
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y <- rep(c(1, 2, 3), each = 8)

  result <- moda_full(X, y, ncomp = 3,
                      numClusters = 2,
                      pcaFirst = TRUE,
                      pcaVar = 0.90,
                      B_init = "pca",
                      maxIter = 15,
                      verbose = FALSE)

  expect_s3_class(result, "moda_full")
  expect_s3_class(result, "discriminant_projector")

  expect_equal(dim(result$v), c(d, 3))

  expect_length(result$clusters, 3)
  for (cls in seq_along(result$clusters)) {
    clIDs <- unique(result$clusters[[cls]])
    expect_true(all(clIDs %in% c(1, 2)))
  }

  expect_true(!is.null(result$pcaInfo))
  expect_named(result$pcaInfo, c("U", "mean"))

  expect_gt(length(result$objVals), 0)
  expect_false(anyNA(result$objVals))
})

test_that("moda_full throws error when y has only one class", {
  d <- 5
  n <- 10
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y <- rep(1, n)

  expect_error(
    moda_full(X, y, ncomp = 2, numClusters = 1),
    "There must be at least 2 distinct classes in 'y'."
  )
})

test_that("moda_full throws error if ncomp > dimension after PCA", {
  d <- 8
  n <- 8
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y <- rep(1:2, each = 4)

  expect_error(
    moda_full(X, y, ncomp = 10, pcaFirst = TRUE),
    "cannot exceed the dimension after PCA"
  )
})
