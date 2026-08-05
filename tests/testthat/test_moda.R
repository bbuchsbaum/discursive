moda_covariance_matrix <- function(model) {
  model$U %*% model$Lambda %*% t(model$U) +
    model$sigma2 * diag(nrow(model$U))
}

moda_objective_oracle <- function(B, covariance_models, mean_models) {
  total <- 0
  for (i in seq_along(covariance_models)) {
    for (r1 in seq_along(covariance_models[[i]])) {
      Sigma_i <- moda_covariance_matrix(covariance_models[[i]][[r1]])
      denominator <- crossprod(B, Sigma_i %*% B)
      inverse <- solve(denominator)
      mu_i <- mean_models[[i]][[r1]]
      for (j in seq_along(covariance_models)) {
        if (j == i) next
        for (r2 in seq_along(covariance_models[[j]])) {
          Sigma_j <- moda_covariance_matrix(covariance_models[[j]][[r2]])
          delta <- mu_i - mean_models[[j]][[r2]]
          numerator <- crossprod(B, (tcrossprod(delta) + Sigma_j) %*% B)
          total <- total + sum(inverse * numerator)
        }
      }
    }
  }
  -total
}

moda_fixture <- function() {
  set.seed(320)
  y <- factor(rep(c("a", "b", "c"), each = 12))
  X <- matrix(rnorm(length(y) * 5, sd = 0.45), ncol = 5)
  X[y == "a", 1:2] <- X[y == "a", 1:2] - 1
  X[y == "b", 2:3] <- X[y == "b", 2:3] + 1
  X[y == "c", 4:5] <- X[y == "c", 4:5] - 1
  colnames(X) <- paste0("x", 1:5)
  list(X = X, y = y)
}

test_that("MODA uses the package-wide row-sample projector contract", {
  dat <- moda_fixture()
  fit <- moda_full(
    dat$X, dat$y, k = 2, numClusters = 1,
    pcaFirst = FALSE, B_init = "pca", maxIter = 8
  )

  expect_s3_class(fit, "moda")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(dim(fit$v), c(ncol(dat$X), 2))
  expect_equal(fit$B, fit$v, tolerance = 0)
  expect_equal(unname(crossprod(fit$v)), diag(2), tolerance = 1e-10)
  expect_equal(multivarious::project(fit, dat$X), fit$s, tolerance = 1e-12)
  expect_equal(length(fit$labels), nrow(dat$X))
  expect_error(
    moda_full(t(dat$X), dat$y, k = 2),
    "one label per row"
  )
})

test_that("MODA objective history is monotone and matches a literal oracle", {
  dat <- moda_fixture()
  fit <- moda_full(
    dat$X, dat$y, k = 1, numClusters = 1,
    pcaFirst = FALSE, B_init = "pca", maxIter = 12, tol = 1e-8
  )
  oracle <- moda_objective_oracle(
    fit$v, fit$cluster_covariances, fit$cluster_means
  )

  expect_true(all(is.finite(fit$objective_history)))
  expect_true(all(diff(fit$objective_history) <= 1e-10))
  expect_equal(tail(fit$objective_history, 1), oracle, tolerance = 1e-8)
  expect_equal(length(fit$objective_history), fit$iterations + 1L)
  expect_equal(fit$objVals, fit$objective_history[-1L])
})

test_that("MODA is translation invariant under fitted centering", {
  dat <- moda_fixture()
  shift <- c(100, -30, 20, 5, -10)
  shifted_X <- sweep(dat$X, 2L, shift, FUN = "+")
  fit <- moda_full(
    dat$X, dat$y, k = 2, numClusters = 1,
    pcaFirst = TRUE, B_init = "pca", maxIter = 8
  )
  shifted <- moda_full(
    shifted_X, dat$y, k = 2, numClusters = 1,
    pcaFirst = TRUE, B_init = "pca", maxIter = 8
  )

  expect_equal(tcrossprod(fit$v), tcrossprod(shifted$v), tolerance = 1e-8)
  expect_equal(as.matrix(dist(fit$s)), as.matrix(dist(shifted$s)), tolerance = 1e-8)
  expect_equal(fit$objective_history, shifted$objective_history, tolerance = 1e-7)
})

test_that("MODA local seed is reproducible without clobbering caller RNG", {
  dat <- moda_fixture()
  set.seed(887)
  before <- .Random.seed
  fit1 <- moda_full(
    dat$X, dat$y, k = 2, numClusters = 2,
    pcaFirst = FALSE, B_init = "random", maxIter = 5, seed = 14
  )
  after <- .Random.seed
  fit2 <- moda_full(
    dat$X, dat$y, k = 2, numClusters = 2,
    pcaFirst = FALSE, B_init = "random", maxIter = 5, seed = 14
  )

  expect_identical(after, before)
  expect_identical(.Random.seed, before)
  expect_equal(fit1$v, fit2$v, tolerance = 0)
  expect_equal(fit1$clusters, fit2$clusters)
  expect_equal(fit1$objective_history, fit2$objective_history, tolerance = 0)
})

test_that("deterministic MODA paths do not touch RNG", {
  dat <- moda_fixture()
  set.seed(991)
  before <- .Random.seed
  invisible(moda_full(
    dat$X, dat$y, k = 1, numClusters = 1,
    pcaFirst = FALSE, B_init = "pca", maxIter = 3
  ))
  expect_identical(.Random.seed, before)
})

test_that("MODA validates dimensions, labels, and optimization parameters", {
  dat <- moda_fixture()
  expect_error(moda_full(dat$X, dat$y[-1], k = 2), "one label per row")
  expect_error(moda_full(dat$X, rep("a", nrow(dat$X)), k = 2), "at least two")
  expect_error(moda_full(dat$X, dat$y, k = 0), "'k'")
  expect_error(moda_full(dat$X, dat$y, k = 2, numClusters = 13), "numClusters")
  expect_error(moda_full(dat$X, dat$y, k = 2, pcaVar = 0), "pcaVar")
  expect_error(moda_full(dat$X, dat$y, k = 2, maxIter = 0), "maxIter")
  expect_error(moda_full(dat$X, dat$y, k = 2, tol = 0), "tol")
  expect_error(moda_full(dat$X, dat$y, k = 2, B_init = "bad"), "arg")
})
