pca_lda_fixture <- function() {
  set.seed(1901)
  y <- factor(c(rep("a", 4), rep("b", 8), rep("c", 14)))
  means <- rbind(
    c(-1.5, 0.5, 0, 0),
    c(1, -0.5, 0.5, 0),
    c(0, 1.5, -1, 0.5)
  )
  X <- matrix(rnorm(length(y) * 4, sd = 0.6), ncol = 4)
  X <- X + means[as.integer(y), ]
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  list(X = X, y = y)
}

pca_lda_scatter_reference <- function(X, y) {
  y <- droplevels(factor(y))
  n <- nrow(X)
  counts <- tabulate(y, nbins = nlevels(y))
  means <- rowsum(X, y, reorder = TRUE) / counts
  residuals <- X - means[as.integer(y), , drop = FALSE]
  offsets <- sweep(means, 2, colMeans(X), "-")
  weighted_offsets <- sweep(offsets, 1, sqrt(counts / n), "*")

  list(
    Sw = crossprod(residuals) / n,
    Sb = crossprod(weighted_offsets)
  )
}

pca_lda_projection_matrix <- function(A) {
  Q <- qr.Q(qr(A))
  tcrossprod(Q)
}

literal_pca_lda <- function(X, y, dp, dl) {
  pca <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = dp)
  pca_vectors <- pca$rotation[, seq_len(dp), drop = FALSE]
  pca_scores <- sweep(X, 2, colMeans(X), "-") %*% pca_vectors
  scatter <- pca_lda_scatter_reference(pca_scores, y)
  direct <- eigen(solve(scatter$Sw, scatter$Sb))
  take <- order(Re(direct$values), decreasing = TRUE)[seq_len(dl)]

  list(
    vectors = pca_vectors %*% Re(direct$vectors[, take, drop = FALSE]),
    values = Re(direct$values[take])
  )
}

test_that("PCA-LDA satisfies the Fisher scatter identities", {
  dat <- pca_lda_fixture()
  fit <- pca_lda(dat$X, dat$y)
  scatter <- pca_lda_scatter_reference(dat$X, dat$y)

  expect_s3_class(fit, "pca_lda")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(c(fit$dp, fit$di, fit$dl), c(4L, 4L, 2L))
  expect_equal(dim(fit$v), c(4, 2))
  expect_equal(
    unname(crossprod(fit$v, scatter$Sw %*% fit$v)),
    diag(2),
    tolerance = 1e-9
  )
  expect_equal(
    unname(crossprod(fit$v, scatter$Sb %*% fit$v)),
    diag(unname(fit$eigenvalues)),
    tolerance = 1e-9
  )

  residual <- scatter$Sb %*% fit$v -
    scatter$Sw %*% fit$v %*% diag(unname(fit$eigenvalues))
  expect_lt(max(abs(residual)), 1e-8)
})

test_that("PCA-LDA matches direct and MASS LDA oracles without truncation", {
  dat <- pca_lda_fixture()
  fit <- pca_lda(dat$X, dat$y)
  literal <- literal_pca_lda(dat$X, dat$y, dp = 4, dl = 2)
  mass_fit <- MASS::lda(x = dat$X, grouping = dat$y)

  expect_equal(
    pca_lda_projection_matrix(fit$v),
    pca_lda_projection_matrix(literal$vectors),
    tolerance = 1e-8
  )
  expect_equal(
    pca_lda_projection_matrix(fit$v),
    pca_lda_projection_matrix(mass_fit$scaling),
    tolerance = 1e-8
  )
  expect_equal(unname(fit$eigenvalues), literal$values, tolerance = 1e-8)
})

test_that("truncated PCA-LDA matches a literal PCA then LDA calculation", {
  dat <- pca_lda_fixture()
  fit <- pca_lda(dat$X, dat$y, dp = 3, di = 3, dl = 2)
  literal <- literal_pca_lda(dat$X, dat$y, dp = 3, dl = 2)

  expect_equal(
    pca_lda_projection_matrix(fit$v),
    pca_lda_projection_matrix(literal$vectors),
    tolerance = 1e-8
  )
  expect_equal(unname(fit$eigenvalues), literal$values, tolerance = 1e-8)
})

test_that("PCA-LDA projection metadata and preprocessing are coherent", {
  dat <- pca_lda_fixture()
  fit <- pca_lda(dat$X, dat$y, dl = 1)
  fit_alias <- pca_lda(dat$X, dat$y, ncomp = 1)

  expect_equal(
    pca_lda_projection_matrix(fit_alias$v),
    pca_lda_projection_matrix(fit$v),
    tolerance = 1e-10
  )
  expect_error(
    pca_lda(dat$X, dat$y, dl = 1, ncomp = 2),
    "compatibility alias"
  )

  expect_equal(
    multivarious::project(fit, dat$X),
    fit$s,
    tolerance = 1e-10
  )
  expect_equal(
    dim(multivarious::project(fit, dat$X[1:3, , drop = FALSE])),
    c(3, 1)
  )
  expect_equal(unname(fit$sdev), stats::sd(fit$s[, 1]), tolerance = 1e-12)
  expect_equal(colnames(fit$v), "PCALDA1")
  expect_equal(rownames(fit$v), colnames(dat$X))
  expect_equal(unname(fit$pca_center), rep(0, ncol(dat$X)), tolerance = 1e-12)

  fit_pass <- pca_lda(dat$X, dat$y, preproc = multivarious::pass())
  expect_equal(fit_pass$pca_center, colMeans(dat$X), tolerance = 1e-12)
  expect_equal(
    multivarious::project(fit_pass, dat$X),
    fit_pass$s,
    tolerance = 1e-10
  )
})

test_that("PCA-LDA handles one feature and undersampled high-dimensional data", {
  X_one <- matrix(c(-3, -2, -1, 1, 2, 4), ncol = 1)
  y_one <- factor(rep(c("left", "right"), each = 3))
  fit_one <- pca_lda(X_one, y_one)

  expect_equal(dim(fit_one$v), c(1, 1))
  expect_equal(c(fit_one$dp, fit_one$di, fit_one$dl), c(1L, 1L, 1L))
  expect_true(all(is.finite(fit_one$v)))

  set.seed(1902)
  y_high <- factor(rep(1:3, each = 4))
  X_high <- matrix(rnorm(length(y_high) * 30), ncol = 30)
  X_high[y_high == 2, 1:3] <- X_high[y_high == 2, 1:3] + 1
  X_high[y_high == 3, 4:6] <- X_high[y_high == 3, 4:6] - 1
  fit_high <- pca_lda(X_high, y_high)

  expect_equal(fit_high$dp, nrow(X_high) - nlevels(y_high))
  expect_equal(fit_high$di, fit_high$dp)
  expect_equal(fit_high$dl, 2L)
  expect_equal(dim(fit_high$v), c(30, 2))
  expect_true(all(is.finite(fit_high$v)))
  expect_true(all(is.finite(fit_high$s)))
})

test_that("PCA-LDA caps automatic dimensions at numerical ranks", {
  set.seed(1903)
  y <- factor(rep(1:3, each = 4))
  base <- matrix(rnorm(length(y) * 2), ncol = 2)
  base[y == 2, 1] <- base[y == 2, 1] + 1
  base[y == 3, 2] <- base[y == 3, 2] - 1
  X <- cbind(base, base[, 1], base[, 2], rowSums(base))
  fit <- pca_lda(X, y)

  expect_equal(fit$pca_rank, 2L)
  expect_equal(fit$dp, 2L)
  expect_lte(fit$di, fit$dp)
  expect_lte(fit$dl, 2L)
  expect_true(all(is.finite(fit$v)))
})

test_that("PCA-LDA is invariant to replication and common predictor scaling", {
  dat <- pca_lda_fixture()
  fit <- pca_lda(dat$X, dat$y)
  fit_replicated <- pca_lda(rbind(dat$X, dat$X), factor(rep(dat$y, 2)))
  scale_factor <- 1e-9
  fit_scaled <- pca_lda(dat$X * scale_factor, dat$y)

  expect_equal(fit_replicated$v, fit$v, tolerance = 1e-8)
  expect_equal(
    fit_replicated$s[seq_len(nrow(dat$X)), , drop = FALSE],
    fit$s,
    tolerance = 1e-8
  )
  expect_equal(fit_replicated$eigenvalues, fit$eigenvalues, tolerance = 1e-9)
  expect_equal(fit_scaled$v * scale_factor, fit$v, tolerance = 1e-7)
  expect_equal(fit_scaled$s, fit$s, tolerance = 1e-8)
  expect_equal(fit_scaled$eigenvalues, fit$eigenvalues, tolerance = 1e-8)
})

test_that("randomized PCA-LDA fits preserve whitening and finite outputs", {
  for (seed in 1:5) {
    set.seed(seed)
    y <- factor(rep(1:3, each = 7))
    X <- matrix(rnorm(length(y) * 5), ncol = 5)
    X[y == 2, 1:2] <- X[y == 2, 1:2] + 0.6
    X[y == 3, 3:4] <- X[y == 3, 3:4] - 0.6
    fit <- pca_lda(X, y)
    scatter <- pca_lda_scatter_reference(X, y)

    expect_true(all(is.finite(fit$v)), info = paste("seed", seed))
    expect_true(all(is.finite(fit$s)), info = paste("seed", seed))
    expect_equal(
      unname(crossprod(fit$v, scatter$Sw %*% fit$v)),
      diag(fit$dl),
      tolerance = 1e-8,
      info = paste("seed", seed)
    )
  }
})

test_that("PCA-LDA validates dimensions, labels, and degenerate inputs", {
  dat <- pca_lda_fixture()
  X <- dat$X
  y <- dat$y

  expect_error(pca_lda(X, y[-1]), "one non-missing label")
  expect_error(pca_lda(X, rep("one", nrow(X))), "at least two classes")
  expect_error(pca_lda(X, y, dp = 0), "dp")
  expect_error(pca_lda(X, y, dp = 1.5), "integer")
  expect_error(pca_lda(X, y, dp = 3, di = 4), "di")
  expect_error(pca_lda(X, y, dp = 3, di = 2, dl = 3), "dl")
  expect_error(pca_lda(X, y, tol = 0), "tol")

  X_bad <- X
  X_bad[1, 1] <- Inf
  expect_error(pca_lda(X_bad, y), "finite")
  expect_error(
    pca_lda(matrix(1, nrow = 6, ncol = 2), factor(rep(1:2, each = 3))),
    "zero numerical PCA rank"
  )
  expect_error(
    pca_lda(diag(3), factor(1:3)),
    "within-class degree of freedom"
  )

  X_rank_two <- cbind(X[, 1:2], X[, 1], X[, 2])
  expect_error(pca_lda(X_rank_two, y, dp = 4), "numerical PCA rank")

  within_offsets <- rbind(
    c(-0.5, 0), c(0.5, 0), c(0, -0.5), c(0, 0.5)
  )
  X_rank_one_between <- rbind(
    sweep(within_offsets, 2, c(-2, 0), "+"),
    sweep(within_offsets, 2, c(0, 0), "+"),
    sweep(within_offsets, 2, c(2, 0), "+")
  )
  y_three <- factor(rep(c("a", "b", "c"), each = 4))
  expect_error(
    pca_lda(X_rank_one_between, y_three, dp = 2, di = 2, dl = 2),
    "numerical between-class rank"
  )
})
