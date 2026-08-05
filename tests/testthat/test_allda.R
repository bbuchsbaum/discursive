library(testthat)
library(MASS)

generate_test_data <- function(n_per_class = 30, d = 6, n_classes = 3, sep = 2) {
  X <- matrix(0, nrow = n_per_class * n_classes, ncol = d)
  y <- factor(rep(seq_len(n_classes), each = n_per_class))
  means <- matrix(0, nrow = n_classes, ncol = d)

  for (i in seq_len(n_classes)) {
    means[i, ((i - 1) %% d) + 1] <- sep * i
  }

  sigma <- diag(d) * 0.5
  for (i in seq_len(n_classes)) {
    idx <- which(y == i)
    X[idx, ] <- MASS::mvrnorm(n = n_per_class, mu = means[i, ], Sigma = sigma)
  }

  list(X = X, y = y)
}

expect_similarity_contract <- function(S, y, k, tol = 1e-10) {
  y <- as.factor(y)
  cross_class <- outer(y, y, FUN = "!=")

  expect_equal(dim(S), c(length(y), length(y)))
  expect_true(all(is.finite(S)))
  expect_true(all(S >= -tol))
  expect_equal(rowSums(S), rep(1, nrow(S)), tolerance = tol)
  expect_equal(diag(S), rep(0, nrow(S)), tolerance = tol)
  expect_true(all(abs(S[cross_class]) < tol))
  expect_equal(rowSums(S > tol), rep(k, nrow(S)))
}

subspace_projector <- function(W) {
  Q <- qr.Q(qr(W))
  tcrossprod(Q)
}

test_that("allda returns a valid discriminant projector", {
  set.seed(123)
  dat <- generate_test_data(n_per_class = 20, d = 6, n_classes = 3, sep = 2.5)
  X <- dat$X
  y <- dat$y

  fit <- suppressWarnings(
    allda(X, y, ncomp = 2, k = 5, r = 2, max_iter = 20, tol = 1e-3, var_retained = 0.98)
  )

  expect_s3_class(fit, "allda")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(dim(fit$W), c(ncol(X), 2))
  expect_equal(dim(fit$scores), c(nrow(X), 2))
  expect_equal(dim(fit$S), c(nrow(X), nrow(X)))
  expect_equal(fit$ncomp, 2)
  expect_equal(fit$k, 5)
})

test_that("allda validates key input arguments", {
  set.seed(456)
  dat <- generate_test_data(n_per_class = 10, d = 5, n_classes = 2)
  X <- dat$X
  y <- dat$y

  expect_error(allda(X, y, ncomp = 0, k = 3), "ncomp must be a finite positive integer")
  expect_error(allda(X, y, ncomp = 1, k = 0), "k must be a finite positive integer less than the number of samples")
  expect_error(allda(X, y, ncomp = 1, k = 10), "k must be less than the size of every class")
  expect_error(allda(X, y, ncomp = 1, k = 3, r = 1), "r must be a finite numeric scalar greater than 1")
  expect_error(allda(X, y, ncomp = 1, k = 3, var_retained = 1.1), "var_retained must be a finite numeric scalar between 0 and 1")
  expect_error(allda(X, y, ncomp = 1, k = 3, r = Inf), "r must be a finite numeric scalar greater than 1")
  expect_error(allda(X, y, ncomp = 1, k = 3, tol = Inf), "tol must be a finite positive numeric scalar")
  expect_error(allda(X, y, ncomp = 1, k = 3, max_iter = Inf), "max_iter must be a finite positive integer")
  expect_error(allda(X, y, ncomp = c(1, 2), k = 3), "ncomp must be a finite positive integer")

  X_nonfinite <- X
  X_nonfinite[1, 1] <- Inf
  expect_error(allda(X_nonfinite, y, ncomp = 1, k = 3), "X must contain only finite values")

  X_constant <- matrix(1, nrow = nrow(X), ncol = ncol(X))
  expect_error(
    allda(X_constant, y, ncomp = 1, k = 3),
    "non-zero variation after preprocessing"
  )

  expect_s3_class(
    suppressWarnings(allda(as.data.frame(X), y, ncomp = 1, k = 3, max_iter = 5)),
    "allda"
  )
})

test_that("apply_pca accepts numeric data frames", {
  set.seed(101)
  X <- matrix(rnorm(40), nrow = 10)
  pca_fit <- apply_pca(as.data.frame(X), var_retained = 0.9)

  expect_true(is.matrix(pca_fit$X_pca))
  expect_equal(nrow(pca_fit$X_pca), nrow(X))
})

test_that("ALLDA graph construction is restricted to same-class neighbors", {
  X <- matrix(
    c(
      0.00, 0,
      10.0, 0,
      0.01, 0,
      10.01, 0
    ),
    ncol = 2,
    byrow = TRUE
  )
  y <- factor(c("a", "a", "b", "b"))

  S <- initialize_parameters(X, y, ncomp = 1, k = 1)$S

  expect_equal(S[1, 2], 1)
  expect_equal(S[1, 3], 0)
  expect_true(all(S[y == "a", y == "b"] == 0))
  expect_true(all(S[y == "b", y == "a"] == 0))
  expect_similarity_contract(S, y, k = 1)
})

test_that("update_similarity uses squared distances for adaptive weights", {
  X <- matrix(
    c(
      0, 0,
      1, 0,
      2, 0,
      10, 0,
      11, 0,
      12, 0
    ),
    ncol = 2,
    byrow = TRUE
  )
  y <- factor(rep(c("a", "b"), each = 3))
  W <- matrix(c(1, 0), ncol = 1)

  S <- update_similarity(X, W, y, k = 2, r = 2, epsilon = 0)

  expect_equal(S[1, 2], 0.8, tolerance = 1e-8)
  expect_equal(S[1, 3], 0.2, tolerance = 1e-8)
  expect_equal(rowSums(S), rep(1, nrow(S)), tolerance = 1e-10)
  expect_true(all(S[y == "a", y == "b"] == 0))
  expect_true(all(S[y == "b", y == "a"] == 0))
  expect_similarity_contract(S, y, k = 2)
})

test_that("update_similarity is invariant to global changes of units", {
  X <- matrix(
    c(
      0, 0,
      1, 0,
      3, 0,
      10, 0,
      11, 0,
      13, 0
    ),
    ncol = 2,
    byrow = TRUE
  )
  y <- factor(rep(c("a", "b"), each = 3))
  W <- matrix(c(1, 0), ncol = 1)

  S <- update_similarity(X, W, y, k = 2, r = 2)
  S_small_units <- update_similarity(X * 1e-6, W, y, k = 2, r = 2)

  expect_equal(S_small_units, S, tolerance = 1e-12)
})

test_that("duplicate neighbors receive the limiting normalized weight", {
  X <- matrix(
    c(
      0, 0,
      0, 0,
      1, 0,
      10, 0,
      10, 0,
      11, 0
    ),
    ncol = 2,
    byrow = TRUE
  )
  y <- factor(rep(c("a", "b"), each = 3))
  W <- matrix(c(1, 0), ncol = 1)

  S <- update_similarity(X, W, y, k = 2, r = 2)

  expect_equal(S[1, 2], 1)
  expect_equal(S[1, 3], 0)
  expect_equal(rowSums(S), rep(1, nrow(S)), tolerance = 1e-12)
})

test_that("same-class KNN drops self-neighbors even with duplicate rows", {
  X <- matrix(
    c(
      0, 0,
      0, 0,
      1, 0,
      10, 0,
      10, 0,
      11, 0
    ),
    ncol = 2,
    byrow = TRUE
  )
  y <- factor(rep(c("a", "b"), each = 3))
  W <- diag(2)

  S_init <- initialize_parameters(X, y, ncomp = 1, k = 1)$S
  S_update <- update_similarity(X, W, y, k = 1, r = 2)

  expect_true(all(diag(S_init) == 0))
  expect_true(all(diag(S_update) == 0))
})

test_that("update_W solves a symmetric generalized eigenproblem", {
  set.seed(202)
  X <- matrix(rnorm(48), nrow = 12, ncol = 4)
  y <- factor(rep(1:3, each = 4))
  S <- initialize_parameters(X, y, ncomp = 2, k = 2)$S
  L_A <- update_laplacian(S)
  reg <- 1e-4

  W <- update_W(X, L_A, ncomp = 2, reg = reg)
  ridge <- reg * mean(diag(crossprod(X)))
  St_reg <- crossprod(X) + ridge * diag(ncol(X))

  expect_type(W, "double")
  expect_false(any(Im(W) != 0))
  expect_equal(dim(W), c(ncol(X), 2))
  expect_equal(crossprod(W, St_reg %*% W), diag(2), tolerance = 1e-7)
})

test_that("update_laplacian satisfies graph Laplacian invariants", {
  set.seed(303)
  dat <- generate_test_data(n_per_class = 8, d = 5, n_classes = 3)
  S <- initialize_parameters(dat$X, dat$y, ncomp = 2, k = 3)$S
  L_A <- update_laplacian(S)

  expect_equal(L_A, t(L_A), tolerance = 1e-12)
  expect_equal(rowSums(L_A), rep(0, nrow(L_A)), tolerance = 1e-12)

  for (i in 1:10) {
    z <- rnorm(nrow(L_A))
    quadratic_form <- as.numeric(crossprod(z, L_A %*% z))
    expect_gte(quadratic_form, -1e-10)
  }
})

test_that("update_W matches an analytic diagonal generalized eigensolution", {
  scales <- c(1, 2, 3, 4)
  laplacian_diag <- c(4, 1, 3, 2)
  X <- diag(scales)
  L_A <- diag(laplacian_diag)
  reg <- 0.1

  W <- update_W(X, L_A, ncomp = 2, reg = reg)
  ridge <- reg * mean(scales^2)
  St_reg_diag <- scales^2 + ridge
  generalized_values <- scales^2 * laplacian_diag / St_reg_diag
  expected_indices <- order(generalized_values)[1:2]
  expected_abs <- matrix(0, nrow = length(scales), ncol = 2)
  expected_abs[cbind(expected_indices, seq_along(expected_indices))] <-
    1 / sqrt(St_reg_diag[expected_indices])

  expect_equal(abs(W), expected_abs, tolerance = 1e-7)
})

test_that("update_W is scale invariant and cached whitening is equivalent", {
  set.seed(212)
  X <- matrix(rnorm(60), nrow = 15, ncol = 4)
  y <- factor(rep(1:3, each = 5))
  L_A <- update_laplacian(initialize_parameters(X, y, ncomp = 2, k = 2)$S)

  W <- update_W(X, L_A, ncomp = 2, reg = 1e-4)
  W_small_units <- update_W(X * 1e-6, L_A, ncomp = 2, reg = 1e-4)
  whitener <- .prepare_allda_whitener(X, reg = 1e-4)
  W_cached <- update_W(X, L_A, ncomp = 2, reg = 1e-4, whitener = whitener)

  expect_equal(subspace_projector(W_small_units), subspace_projector(W), tolerance = 1e-8)
  expect_equal(W_cached, W, tolerance = 1e-10)
})

test_that("allda is invariant to global changes of measurement units", {
  set.seed(929)
  dat <- generate_test_data(n_per_class = 12, d = 6, n_classes = 3, sep = 2)

  fit <- suppressWarnings(
    allda(dat$X, dat$y, ncomp = 2, k = 4, max_iter = 8, reg = 1e-4)
  )
  fit_small_units <- suppressWarnings(
    allda(dat$X * 1e-6, dat$y, ncomp = 2, k = 4, max_iter = 8, reg = 1e-4)
  )

  expect_equal(fit_small_units$S, fit$S, tolerance = 1e-10)
  expect_equal(
    subspace_projector(fit_small_units$W),
    subspace_projector(fit$W),
    tolerance = 1e-8
  )
})

test_that("allda prepares fixed whitening only once", {
  set.seed(930)
  dat <- generate_test_data(n_per_class = 8, d = 5, n_classes = 3, sep = 2)
  original_prepare <- discursive:::.prepare_allda_whitener
  prepare_calls <- 0L

  testthat::local_mocked_bindings(
    .prepare_allda_whitener = function(...) {
      prepare_calls <<- prepare_calls + 1L
      original_prepare(...)
    },
    .package = "discursive"
  )

  suppressWarnings(
    allda(dat$X, dat$y, ncomp = 2, k = 3, max_iter = 3, reg = 1e-4)
  )

  expect_equal(prepare_calls, 1L)
})

test_that("update_similarity is permutation equivariant", {
  set.seed(404)
  dat <- generate_test_data(n_per_class = 6, d = 5, n_classes = 3)
  W <- matrix(rnorm(ncol(dat$X) * 2), ncol = 2)
  S <- update_similarity(dat$X, W, dat$y, k = 2, r = 2)

  permutation <- sample(seq_len(nrow(dat$X)))
  S_permuted <- update_similarity(
    dat$X[permutation, , drop = FALSE],
    W,
    dat$y[permutation],
    k = 2,
    r = 2
  )

  expect_equal(S_permuted, S[permutation, permutation], tolerance = 1e-10)
})

test_that("randomized similarity updates preserve core graph invariants", {
  for (seed in 1:5) {
    set.seed(seed)
    dat <- generate_test_data(n_per_class = 5, d = 4, n_classes = 3, sep = 1.5)
    W <- matrix(rnorm(ncol(dat$X) * 2), ncol = 2)
    S <- update_similarity(dat$X, W, dat$y, k = 2, r = 1.5 + runif(1))

    expect_similarity_contract(S, dat$y, k = 2)
  }
})

test_that("allda remains finite on near-collinear high-condition data", {
  set.seed(505)
  y <- factor(rep(1:3, each = 6))
  latent <- matrix(rnorm(length(y) * 3), ncol = 3)
  class_offsets <- model.matrix(~ y - 1) %*% matrix(
    c(
      0, 0, 0,
      2, 0, 0,
      0, 2, 0
    ),
    nrow = 3,
    byrow = TRUE
  )
  latent <- latent + class_offsets
  X <- cbind(
    latent[, 1],
    latent[, 1] + rnorm(length(y), sd = 1e-8),
    latent[, 2],
    latent[, 2] + rnorm(length(y), sd = 1e-8),
    latent[, 3],
    rowSums(latent) + rnorm(length(y), sd = 1e-8)
  )

  fit <- suppressWarnings(
    allda(X, y, ncomp = 2, k = 3, max_iter = 4, var_retained = 1, reg = 1e-3)
  )

  expect_true(all(is.finite(fit$W)))
  expect_true(all(is.finite(fit$scores)))
  expect_similarity_contract(fit$S, y, k = 3, tol = 1e-8)
})

test_that("allda preprocessing and predict are coherent", {
  set.seed(789)
  dat <- generate_test_data(n_per_class = 15, d = 5, n_classes = 3, sep = 2)
  X <- dat$X
  y <- dat$y

  fit_center <- suppressWarnings(allda(X, y, ncomp = 2, k = 4, max_iter = 10))
  Xp_center <- multivarious::transform(fit_center$preproc, X)
  expect_true(all(abs(colMeans(Xp_center)) < 1e-8))

  expect_error(
    allda(X, y, ncomp = 2, k = 4, preproc = multivarious::pass(), max_iter = 10),
    "pre-centered"
  )

  X_centered <- sweep(X, 2, colMeans(X), FUN = "-")
  fit_pass <- suppressWarnings(
    allda(X_centered, y, ncomp = 2, k = 4, preproc = multivarious::pass(), max_iter = 10)
  )
  Xp_pass <- multivarious::transform(fit_pass$preproc, X_centered)
  expect_equal(Xp_pass, X_centered, tolerance = 1e-10)

  translation <- seq_len(ncol(X)) * 100
  X_translated <- sweep(X, 2, translation, FUN = "+")
  fit_translated <- suppressWarnings(allda(X_translated, y, ncomp = 2, k = 4, max_iter = 10))
  expect_equal(
    subspace_projector(fit_translated$W),
    subspace_projector(fit_center$W),
    tolerance = 1e-7
  )

  pred_train <- multivarious::project(fit_center, X)
  expect_equal(pred_train, fit_center$scores, tolerance = 1e-8)

  X_new <- X[1:10, , drop = FALSE]
  pred_new <- multivarious::project(fit_center, X_new)
  expect_equal(dim(pred_new), c(10, fit_center$ncomp))
})

test_that("allda returns a projection coherent with its final graph", {
  set.seed(818)
  dat <- generate_test_data(n_per_class = 12, d = 6, n_classes = 3, sep = 2)

  fit <- suppressWarnings(
    allda(dat$X, dat$y, ncomp = 2, k = 4, max_iter = 1, tol = 1e-12, reg = 1e-4)
  )
  expected_W <- update_W(
    fit$pca_result$X_pca,
    update_laplacian(fit$S),
    ncomp = 2,
    reg = 1e-4
  )

  expect_false(fit$converged)
  expect_true(is.finite(fit$final_delta))
  expect_true(is.finite(fit$objective))
  expect_equal(
    subspace_projector(fit$W_pca),
    subspace_projector(expected_W),
    tolerance = 1e-8
  )
})
