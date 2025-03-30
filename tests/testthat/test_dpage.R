# tests/testthat/test-dgpage.R
library(testthat)

test_that("dgpage_discriminant works with random data", {
  set.seed(123)
  n <- 20
  d <- 5
  K <- 2
  
  # Simulate data from 2 classes
  X <- matrix(rnorm(n * d), nrow=n, ncol=d)
  y <- factor(rep(1:K, length.out=n))
  
  # Basic usage
  dp <- dgpage_discriminant(X, y, r=2, alpha=1e-3, beta=1e-4,
                            maxiter=5, verbose=FALSE)
  
  # Check classes
  expect_s3_class(dp, "dgpage_projector")
  expect_s3_class(dp, "discriminant_projector")
  
  # Check dimensions
  expect_equal(dim(dp$v), c(d, 2))      # loadings
  expect_equal(dim(dp$s), c(n, 2))      # scores
  expect_equal(length(dp$sdev), 2)
  
  # Predict
  preds_1nn <- predict(dp, X, method="1nn")
  expect_length(preds_1nn, n)
  expect_s3_class(preds_1nn, "factor")
  
  # LDA approach
  preds_lda <- predict(dp, X, method="lda", type="class")
  expect_length(preds_lda, n)
  expect_s3_class(preds_lda, "factor")
  
  # Probability approach
  probs_lda <- predict(dp, X, method="lda", type="prob")
  expect_equal(dim(probs_lda), c(n, K))
})

test_that("dgpage_discriminant works on iris data", {
  skip_if_not_installed("datasets")
  data("iris")
  
  # Use only 2 classes to keep it simpler, or all 3
  # Let's do all 3 classes but a smaller sample
  idx <- c(1:40, 51:90, 101:140)  # 120 samples
  X   <- as.matrix(iris[idx, 1:4])
  y   <- factor(iris[idx, 5])
  
  set.seed(999)
  dp <- dgpage_discriminant(X, y, r=2, alpha=1e-3, beta=1e-4,
                            maxiter=5, verbose=FALSE)
  
  expect_s3_class(dp, "dgpage_projector")
  expect_s3_class(dp, "discriminant_projector")
  
  # Quick dimension checks
  expect_equal(ncol(dp$s), 2)
  expect_equal(nrow(dp$s), length(y))
  
  # Basic prediction
  preds <- predict(dp, new_data=X, method="1nn")
  expect_length(preds, length(y))
  
  # We can get a crude accuracy check:
  acc <- mean(preds == y)
  # It's random and with only 5 iterations, but let's at least expect > 0.4
  expect_gt(acc, 0.4)
  
  # LDA-based prediction
  preds_lda <- predict(dp, new_data=X, method="lda", type="class")
  expect_s3_class(preds_lda, "factor")
  acc_lda <- mean(preds_lda == y)
  expect_gt(acc_lda, 0.4)  # again, just a sanity check
})