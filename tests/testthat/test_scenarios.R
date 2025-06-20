library(testthat)

# helper to generate simple gaussian data with specified class means
sim_data <- function(n_per_class, d, means, sd = 1) {
  k <- length(n_per_class)
  total <- sum(n_per_class)
  X <- matrix(NA, nrow = total, ncol = d)
  y <- integer(total)
  idx <- 1
  for (i in seq_len(k)) {
    rows <- idx:(idx + n_per_class[i] - 1)
    X[rows, ] <- matrix(rnorm(length(rows) * d, mean = means[i], sd = sd), ncol = d)
    y[rows] <- i
    idx <- idx + n_per_class[i]
  }
  list(X = X, y = factor(y))
}

# list of methods to test
method_fns <- list(
  olda = function(X, y) olda(X, y),
  ulda = function(X, y) ulda(X, y),
  pca_lda = function(X, y, dp, di, dl) pca_lda(X, y, dp = dp, di = di, dl = dl),
  gmmsd = function(X, y, dim) gmmsd(X, y, dim = dim),
  fastolda = function(X, y) fastolda(X, y)
)

check_projector <- function(dp, n) {
  expect_s3_class(dp, "discriminant_projector")
  expect_equal(nrow(dp$s), n)
  expect_equal(ncol(dp$s), ncol(dp$v))
  expect_length(dp$sdev, ncol(dp$v))
}

# Scenario 1: balanced classes with moderate separation

test_that("balanced gaussian classes", {
  set.seed(100)
  dat <- sim_data(rep(20, 3), d = 6, means = c(0, 3, 6))
  res <- list(
    olda = method_fns$olda(dat$X, dat$y),
    ulda = method_fns$ulda(dat$X, dat$y),
    pca_lda = method_fns$pca_lda(dat$X, dat$y, dp = 5, di = 4, dl = 2),
    gmmsd = method_fns$gmmsd(dat$X, dat$y, dim = 2),
    fastolda = method_fns$fastolda(dat$X, dat$y)
  )
  lapply(res, check_projector, n = sum(rep(20,3)))
})

# Scenario 2: high-dimensional, p > n

test_that("high dimensional small sample", {
  set.seed(200)
  dat <- sim_data(rep(6, 3), d = 40, means = c(0, 1, 2))
  res <- list(
    olda = method_fns$olda(dat$X, dat$y),
    ulda = method_fns$ulda(dat$X, dat$y),
    pca_lda = method_fns$pca_lda(dat$X, dat$y, dp = 15, di = 5, dl = 2),
    gmmsd = method_fns$gmmsd(dat$X, dat$y, dim = 2),
    fastolda = method_fns$fastolda(dat$X, dat$y)
  )
  lapply(res, check_projector, n = sum(rep(6,3)))
})

# Scenario 3: imbalanced classes with an outlier

test_that("imbalanced classes with outlier", {
  set.seed(300)
  dat <- sim_data(c(30, 10), d = 8, means = c(0, 2))
  # add an extreme outlier
  dat$X[1, ] <- dat$X[1, ] + 15
  res <- list(
    olda = method_fns$olda(dat$X, dat$y),
    ulda = method_fns$ulda(dat$X, dat$y),
    pca_lda = method_fns$pca_lda(dat$X, dat$y, dp = 6, di = 3, dl = 1),
    gmmsd = method_fns$gmmsd(dat$X, dat$y, dim = 1),
    fastolda = method_fns$fastolda(dat$X, dat$y)
  )
  lapply(res, check_projector, n = sum(c(30,10)))
})

