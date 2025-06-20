library(testthat)

# Test group_means with both standard and single-sample groups

test_that("group_means handles various group sizes", {
  X1 <- matrix(c(1,2, 3,4, 5,6), ncol = 2, byrow = TRUE)
  Y1 <- factor(c("A", "B", "B"))
  gm1 <- group_means(Y1, X1)
  expected1 <- rbind(A = X1[1,], B = colMeans(X1[2:3,]))
  expect_equal(gm1, expected1)

  X2 <- matrix(c(7,8, 9,10), ncol = 2, byrow = TRUE)
  Y2 <- factor(c("C", "D"))
  gm2 <- group_means(Y2, X2)
  expect_equal(gm2, X2)
  expect_equal(rownames(gm2), c("C", "D"))
})

# Test compute_Htdiff simply mirrors t(diff(X))

test_that("compute_Htdiff matches transposed differences", {
  X <- matrix(1:6, nrow = 3, ncol = 2)
  expect_equal(compute_Htdiff(X), t(diff(X)))
})

# Basic wMLDA functionality with binary weighting

test_that("wMLDA runs with binary weights", {
  skip_if_not_installed("multivarious")
  X <- matrix(c(1,2,3, 4,5,6, 7,8,9, 10,11,12), nrow = 4, byrow = TRUE)
  Y <- matrix(c(1,0,
                1,0,
                0,1,
                0,1), nrow = 4, byrow = TRUE)
  res <- wMLDA(X, Y, weight_method = "binary", ncomp = 1, preproc = multivarious::pass())
  expect_s3_class(res, "discriminant_projector")
  expect_equal(dim(res$v), c(ncol(X), 1))
  expect_equal(dim(res$s), c(nrow(X), 1))
  expect_equal(res$labels, Y)
})

