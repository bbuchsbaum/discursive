wmlda_fixture <- function() {
  set.seed(2101)
  Y <- rbind(
    cbind(1, 0, 0)[rep(1, 4), , drop = FALSE],
    cbind(0, 1, 0)[rep(1, 4), , drop = FALSE],
    cbind(0, 0, 1)[rep(1, 4), , drop = FALSE],
    cbind(1, 1, 0)[rep(1, 2), , drop = FALSE],
    cbind(0, 1, 1)[rep(1, 2), , drop = FALSE],
    cbind(1, 0, 1)[rep(1, 2), , drop = FALSE]
  )
  colnames(Y) <- c("red", "green", "blue")
  label_means <- rbind(
    c(-1.5, 0.4, 0.2, 0),
    c(1.2, -0.8, 0.4, 0.5),
    c(0.2, 1.4, -1.0, -0.4)
  )
  X <- Y %*% label_means / rowSums(Y) +
    matrix(rnorm(nrow(Y) * 4, sd = 0.35), ncol = 4)
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  list(X = X, Y = Y)
}

wmlda_dense_scatter_reference <- function(X, W) {
  instance_weights <- rowSums(W)
  class_weights <- colSums(W)
  total_weight <- sum(instance_weights)
  normalized_weights <- sweep(W, 2, sqrt(class_weights), "/")
  within_operator <- diag(instance_weights) -
    tcrossprod(normalized_weights)
  between_operator <- tcrossprod(normalized_weights) -
    tcrossprod(instance_weights) / total_weight

  list(
    Sw = crossprod(X, within_operator %*% X) / total_weight,
    Sb = crossprod(X, between_operator %*% X) / total_weight
  )
}

wmlda_projection_matrix <- function(A) {
  Q <- qr.Q(qr(A))
  tcrossprod(Q)
}

strip_weight_diagnostics <- function(W) {
  attr(W, "iterations") <- NULL
  attr(W, "converged") <- NULL
  attr(W, "objective") <- NULL
  W
}

wmlda_fuzzy_update_reference <- function(X, Y, W) {
  means <- matrix(0, ncol(Y), ncol(X))
  for (k in seq_len(ncol(Y))) {
    squared_weights <- W[, k]^2
    means[k, ] <- colSums(X * squared_weights) / sum(squared_weights)
  }

  updated <- matrix(0, nrow(Y), ncol(Y))
  for (i in seq_len(nrow(Y))) {
    relevant <- which(Y[i, ] == 1)
    distances <- rowSums(
      sweep(means[relevant, , drop = FALSE], 2, X[i, ], "-")^2
    )
    zero_distance <- which(distances == 0)
    if (length(zero_distance)) {
      selected <- relevant[zero_distance]
      updated[i, selected] <- 1 / length(selected)
    } else {
      inverse_distance <- 1 / distances
      updated[i, relevant] <- inverse_distance / sum(inverse_distance)
    }
  }
  updated
}

wmlda_dependence_epoch_reference <- function(X, Y) {
  centered <- sweep(X, 2, colMeans(X), "-")
  theta <- tcrossprod(centered)
  W <- Y / rowSums(Y)
  for (i in sample.int(nrow(Y))) {
    relevant <- which(Y[i, ] == 1)
    others <- setdiff(seq_len(nrow(Y)), i)
    scores <- vapply(relevant, function(k) {
      sum(theta[i, others] * W[others, k])
    }, numeric(1))
    W[i, ] <- 0
    W[i, relevant[which.max(scores)]] <- 1
  }
  W
}

test_that("wMLDA matches the defining dense scatter operators", {
  dat <- wmlda_fixture()
  fit <- wMLDA(dat$X, dat$Y, weight_method = "binary", ncomp = 2)
  Xp <- multivarious::transform(fit$preproc, dat$X)
  scatter <- wmlda_dense_scatter_reference(Xp, fit$weights)
  denominator <- scatter$Sw + diag(fit$ridge, ncol(dat$X))

  expect_s3_class(fit, "wMLDA")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(fit$ncomp, 2L)
  expect_equal(fit$active_labels, colnames(dat$Y))
  expect_equal(fit$total_weight, sum(dat$Y))
  expect_gt(fit$total_weight, nrow(dat$Y))
  expect_equal(
    unname(crossprod(fit$v, denominator %*% fit$v)),
    diag(2),
    tolerance = 1e-8
  )
  expect_equal(
    unname(crossprod(fit$v, scatter$Sb %*% fit$v)),
    diag(unname(fit$eigenvalues)),
    tolerance = 1e-8
  )

  residual <- scatter$Sb %*% fit$v -
    denominator %*% fit$v %*% diag(unname(fit$eigenvalues))
  expect_lt(max(abs(residual)), 1e-7)
})

test_that("wMLDA matches a direct nonsymmetric generalized-eigen oracle", {
  dat <- wmlda_fixture()
  fit <- wMLDA(dat$X, dat$Y, weight_method = "entropy", ncomp = 2)
  Xp <- multivarious::transform(fit$preproc, dat$X)
  scatter <- wmlda_dense_scatter_reference(Xp, fit$weights)
  denominator <- scatter$Sw + diag(fit$ridge, ncol(dat$X))
  direct <- eigen(solve(denominator, scatter$Sb))
  take <- order(Re(direct$values), decreasing = TRUE)[1:2]
  direct_vectors <- Re(direct$vectors[, take, drop = FALSE])

  expect_equal(
    unname(fit$eigenvalues),
    Re(direct$values[take]),
    tolerance = 1e-8
  )
  expect_equal(
    wmlda_projection_matrix(fit$v),
    wmlda_projection_matrix(direct_vectors),
    tolerance = 1e-7
  )
})

test_that("closed-form label weights match independent formulas", {
  dat <- wmlda_fixture()
  Y <- dat$Y

  binary <- discursive:::weight_binary(Y)
  entropy <- discursive:::weight_entropy(Y)
  correlation <- discursive:::weight_correlation(Y)

  expected_correlation <- matrix(0, ncol(Y), ncol(Y))
  for (j in seq_len(ncol(Y))) {
    for (k in seq_len(ncol(Y))) {
      expected_correlation[j, k] <- sum(Y[, j] * Y[, k]) /
        sqrt(sum(Y[, j]^2) * sum(Y[, k]^2))
    }
  }
  expected_correlation <- t(vapply(seq_len(nrow(Y)), function(i) {
    drop(expected_correlation %*% Y[i, ]) / sum(Y[i, ])
  }, numeric(ncol(Y))))

  expect_equal(unname(binary), unname(Y))
  expect_equal(
    unname(entropy),
    unname(Y / rowSums(Y)),
    tolerance = 1e-12
  )
  expect_equal(
    unname(correlation),
    unname(expected_correlation),
    tolerance = 1e-12
  )
  expect_equal(rowSums(entropy), rep(1, nrow(Y)), tolerance = 1e-12)
  expect_true(any(correlation[Y == 0] > 0))
})

test_that("iterative weights respect support, normalization, and objectives", {
  dat <- wmlda_fixture()
  fuzzy <- discursive:::weight_fuzzy(dat$X, dat$Y, max_iter = 100, tol = 1e-8)
  fuzzy_scaled <- discursive:::weight_fuzzy(
    dat$X * 1e-7, dat$Y, max_iter = 100, tol = 1e-8
  )

  expect_true(all(is.finite(fuzzy)))
  expect_true(all(fuzzy >= 0))
  expect_equal(fuzzy[dat$Y == 0], rep(0, sum(dat$Y == 0)))
  expect_equal(rowSums(fuzzy), rep(1, nrow(dat$Y)), tolerance = 1e-12)
  expect_equal(
    strip_weight_diagnostics(fuzzy_scaled),
    strip_weight_diagnostics(fuzzy),
    tolerance = 1e-7
  )
  expect_true(attr(fuzzy, "converged"))
  expect_equal(
    strip_weight_diagnostics(fuzzy),
    wmlda_fuzzy_update_reference(dat$X, dat$Y, fuzzy),
    tolerance = 1e-8
  )
  expect_lte(attr(fuzzy, "iterations"), 100L)
  expect_type(attr(fuzzy, "converged"), "logical")

  set.seed(2102)
  dependence <- discursive:::weight_dependence(dat$X, dat$Y, max_iter = 100)
  objective <- attr(dependence, "objective")
  expect_equal(dependence[dat$Y == 0], rep(0, sum(dat$Y == 0)))
  expect_equal(rowSums(dependence), rep(1, nrow(dat$Y)))
  expect_true(all(dependence %in% c(0, 1)))
  expect_true(all(diff(objective) >= -1e-10))
  expect_lte(attr(dependence, "iterations"), 100L)
})

test_that("dependence weighting matches a literal sequential coordinate epoch", {
  dat <- wmlda_fixture()
  set.seed(2106)
  expected <- wmlda_dependence_epoch_reference(dat$X, dat$Y)
  set.seed(2106)
  observed <- discursive:::weight_dependence(dat$X, dat$Y, max_iter = 1)

  expect_equal(strip_weight_diagnostics(observed), expected, tolerance = 1e-12)
})

test_that("dependence weighting is reproducible without changing caller RNG", {
  dat <- wmlda_fixture()
  set.seed(2103)
  seed_before <- .Random.seed
  fit_one <- wMLDA(
    dat$X, dat$Y, weight_method = "dependence", seed = 47,
    max_iter_dep = 100
  )

  expect_identical(.Random.seed, seed_before)
  fit_two <- wMLDA(
    dat$X, dat$Y, weight_method = "dependence", seed = 47,
    max_iter_dep = 100
  )
  expect_equal(fit_two$weights, fit_one$weights)
  expect_equal(fit_two$weight_objective, fit_one$weight_objective)
  expect_identical(.Random.seed, seed_before)
})

test_that("one-hot binary wMLDA recovers the ordinary LDA subspace", {
  set.seed(2104)
  y <- factor(rep(c("a", "b", "c"), each = 9))
  means <- rbind(c(-1, 0, 0), c(1, 0.5, 0), c(0, -1, 1))
  X <- matrix(rnorm(length(y) * 3, sd = 0.5), ncol = 3) +
    means[as.integer(y), ]
  Y <- model.matrix(~ y - 1)
  fit <- wMLDA(X, Y, weight_method = "binary", reg = 0)
  mass_fit <- MASS::lda(X, grouping = y)

  expect_equal(
    wmlda_projection_matrix(fit$v),
    wmlda_projection_matrix(mass_fit$scaling),
    tolerance = 1e-8
  )
  expect_equal(
    discursive:::weight_binary(Y),
    discursive:::weight_entropy(Y)
  )
})

test_that("wMLDA projection metadata and preprocessing are coherent", {
  dat <- wmlda_fixture()
  fit <- wMLDA(dat$X, dat$Y, weight_method = "correlation", ncomp = 1)

  expect_equal(multivarious::project(fit, dat$X), fit$s, tolerance = 1e-10)
  expect_equal(
    dim(multivarious::project(fit, dat$X[1:3, , drop = FALSE])),
    c(3, 1)
  )
  expect_equal(unname(fit$sdev), stats::sd(fit$s[, 1]), tolerance = 1e-12)
  expect_equal(dim(fit$label_matrix), dim(dat$Y))
  expect_equal(fit$label_matrix, dat$Y)
  expect_equal(colnames(fit$v), "WMLDA1")
  expect_equal(rownames(fit$v), colnames(dat$X))
  expect_equal(levels(fit$labels), colnames(dat$Y))
  expect_equal(fit$weight_method, "correlation")
})

test_that("wMLDA is invariant to replication and common predictor scaling", {
  dat <- wmlda_fixture()
  fit <- wMLDA(dat$X, dat$Y, weight_method = "binary")
  fit_replicated <- wMLDA(
    rbind(dat$X, dat$X), rbind(dat$Y, dat$Y), weight_method = "binary"
  )
  scale_factor <- 1e-7
  fit_scaled <- wMLDA(
    dat$X * scale_factor, dat$Y, weight_method = "binary"
  )

  expect_equal(fit_replicated$v, fit$v, tolerance = 1e-8)
  expect_equal(fit_replicated$eigenvalues, fit$eigenvalues, tolerance = 1e-9)
  expect_equal(fit_replicated$total_weight, 2 * fit$total_weight)
  expect_equal(fit_scaled$v * scale_factor, fit$v, tolerance = 1e-6)
  expect_equal(fit_scaled$s, fit$s, tolerance = 1e-8)
  expect_equal(fit_scaled$eigenvalues, fit$eigenvalues, tolerance = 1e-8)
})

test_that("wMLDA handles one feature and undersampled high-dimensional data", {
  X_one <- matrix(c(-3, -2, -1, 1, 2, 4), ncol = 1)
  Y_one <- cbind(left = rep(c(1, 0), each = 3), right = rep(c(0, 1), each = 3))
  fit_one <- wMLDA(X_one, Y_one)

  expect_equal(dim(fit_one$v), c(1, 1))
  expect_true(all(is.finite(fit_one$v)))

  set.seed(2105)
  Y_high <- diag(3)[rep(1:3, each = 4), ]
  colnames(Y_high) <- paste0("label", 1:3)
  X_high <- matrix(rnorm(nrow(Y_high) * 30), ncol = 30)
  X_high[, 1:3] <- X_high[, 1:3] + Y_high %*%
    rbind(c(-1, 0, 0), c(1, 0, 0), c(0, 1, 0))
  fit_high <- wMLDA(X_high, Y_high)

  expect_equal(dim(fit_high$v), c(30, 2))
  expect_true(all(is.finite(fit_high$v)))
  expect_true(all(is.finite(fit_high$s)))
})

test_that("wMLDA validates data, labels, dimensions, and controls", {
  dat <- wmlda_fixture()
  X <- dat$X
  Y <- dat$Y

  expect_error(wMLDA(matrix(letters[1:12], nrow = 4), Y[1:4, ]), "numeric matrix")
  expect_error(wMLDA(X, Y[-1, ]), "one row per sample")
  expect_error(wMLDA(X, Y[, 1, drop = FALSE]), "at least two labels")

  Y_nonbinary <- Y
  Y_nonbinary[1, 1] <- 0.5
  expect_error(wMLDA(X, Y_nonbinary), "binary values")
  Y_empty_row <- Y
  Y_empty_row[1, ] <- 0
  expect_error(wMLDA(X, Y_empty_row), "Every row")
  Y_empty_column <- cbind(Y, empty = 0)
  expect_error(wMLDA(X, Y_empty_column), "Every column")
  Y_duplicate_names <- Y
  colnames(Y_duplicate_names)[2] <- colnames(Y_duplicate_names)[1]
  expect_error(wMLDA(X, Y_duplicate_names), "unique")

  expect_error(wMLDA(X, Y, ncomp = 0), "ncomp")
  expect_error(wMLDA(X, Y, ncomp = 1.5), "integer")
  expect_error(wMLDA(X, Y, reg = -1), "reg")
  expect_error(wMLDA(X, Y, tol = 0), "tol")
  expect_error(wMLDA(X, Y, tol_fuzzy = 0), "tol_fuzzy")
  expect_error(wMLDA(X, Y, max_iter_fuzzy = 0), "max_iter_fuzzy")
  expect_error(wMLDA(X, Y, max_iter_dep = 1.5), "integer")
  expect_error(wMLDA(X, Y, seed = -1), "seed")

  X_bad <- X
  X_bad[1, 1] <- Inf
  expect_error(wMLDA(X_bad, Y), "finite")
  expect_error(
    wMLDA(matrix(1, nrow = nrow(Y), ncol = 2), Y),
    "zero numerical scale"
  )
})
