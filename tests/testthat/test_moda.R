# tests/testthat/test-moda_full.R

library(testthat)

## We'll assume you've already `source()`d or installed
## the 'moda_full' function and its internal helpers.

test_that("moda_full runs on a small synthetic dataset (random init, single cluster)", {
  set.seed(123)
  d <- 10
  n <- 20
  X <- matrix(rnorm(d * n), nrow = d, ncol = n)
  y <- rep(c(1, 2), each = n/2)
  
  # We test with numClusters = 1 => effectively ODA
  result <- moda_full(X, y, k = 2,
                      numClusters = 1,
                      pcaFirst = FALSE,
                      B_init = "random",
                      maxIter = 10,  # keep it small for test speed
                      verbose = FALSE)
  
  # Basic checks
  expect_type(result, "list")
  expect_s3_class(result, "moda_full")
  expect_named(result, c("B", "objVals", "clusters", "pcaInfo"))
  
  # B should be dimension (d x k) because pcaFirst=FALSE
  expect_equal(dim(result$B), c(d, 2))
  
  # The objective array shouldn't be empty
  expect_true(length(result$objVals) > 0)
  
  # Because we used single cluster, the cluster assignments should all be 1
  # for each class
  for (cl_vec in result$clusters) {
    # each cl_vec is a vector of cluster IDs for that class
    expect_true(all(cl_vec == 1))
  }
})

test_that("moda_full runs with PCA initialization on synthetic data (multiple clusters)", {
  set.seed(123)
  d <- 12
  n <- 24
  X <- matrix(rnorm(d * n), nrow = d, ncol = n)
  y <- rep(c(1, 2, 3), each = 8)  # 3 classes
  
  # We test with 2 clusters for each class => real multimodal scenario
  # Also use PCA for B initialization
  result <- moda_full(X, y, k = 3,
                      numClusters = 2,
                      pcaFirst = TRUE,
                      pcaVar = 0.90,  # keep 90% variance
                      B_init = "pca",
                      maxIter = 15,
                      verbose = FALSE)
  
  # Basic checks
  expect_s3_class(result, "moda_full")
  expect_named(result, c("B", "objVals", "clusters", "pcaInfo"))
  
  # B dimension: if PCA was used, final B is d x k
  # but after PCA, we had dNew <= n. Then B is re-expanded back to original d
  expect_equal(dim(result$B), c(d, 3))
  
  # The cluster assignments:
  # - We expect a list of length = number of unique classes (3 here)
  # - Each entry has cluster IDs in {1,2} (since numClusters=2)
  expect_length(result$clusters, 3)
  for (cls in seq_along(result$clusters)) {
    # each clusters[[cls]] is a vector of length = #samples in that class
    clIDs <- unique(result$clusters[[cls]])
    expect_true(all(clIDs %in% c(1,2)))
  }
  
  # pcaInfo: Because pcaFirst=TRUE, we expect a list with U, mean
  expect_true(!is.null(result$pcaInfo))
  expect_named(result$pcaInfo, c("U","mean"))
  
  # The objective array must have at least 1 iteration value
  expect_gt(length(result$objVals), 0)
  # We can also check it doesn't contain any NA
  expect_false(anyNA(result$objVals))
})

test_that("moda_full throws error when y has only one class", {
  d <- 5
  n <- 10
  X <- matrix(rnorm(d*n), nrow = d, ncol = n)
  y <- rep(1, n) # only one class
  
  # Should error out because we need at least 2 classes
  expect_error(
    moda_full(X, y, k = 2, numClusters = 1),
    "There must be at least 2 distinct classes in 'y'."
  )
})

test_that("moda_full throws error if k > dimension after PCA", {
  d <- 8
  n <- 8
  X <- matrix(rnorm(d*n), nrow = d, ncol = n)
  y <- rep(1:2, each = 4)
  
  # If we do PCA here, dimension after PCA will be <= n=8
  # Try k=10 => should error
  expect_error(
    moda_full(X, y, k = 10, pcaFirst = TRUE),
    "cannot exceed the dimension after PCA"
  )
})