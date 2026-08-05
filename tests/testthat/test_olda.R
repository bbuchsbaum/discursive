olda_fixture <- function() {
  set.seed(1801)
  y <- factor(c(rep("a", 5), rep("b", 6), rep("c", 7)))
  means <- rbind(
    c(-1.5, 0.5, 0, 0),
    c(1, -0.5, 0.5, 0),
    c(0, 1.5, -1, 0.5)
  )
  X <- matrix(rnorm(length(y) * 4, sd = 0.5), ncol = 4)
  X <- X + means[as.integer(y), ]
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  list(X = X, y = y)
}

olda_scatter_reference <- function(X, y) {
  y <- droplevels(factor(y))
  n <- nrow(X)
  Xc <- sweep(X, 2, colMeans(X), "-")
  counts <- tabulate(y, nbins = nlevels(y))
  means <- rowsum(X, y, reorder = TRUE) / counts
  offsets <- sweep(means, 2, colMeans(X), "-")
  Hb <- sweep(t(offsets), 2, sqrt(counts / n), "*")

  list(
    St = crossprod(Xc) / n,
    Sb = tcrossprod(Hb)
  )
}

olda_projection_matrix <- function(A) {
  Q <- qr.Q(qr(A))
  tcrossprod(Q)
}

test_that("OLDA returns an orthonormal basis of the ULDA subspace", {
  dat <- olda_fixture()
  fit <- olda(dat$X, dat$y)
  source <- ulda(dat$X, dat$y)

  expect_s3_class(fit, "olda")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(fit$ncomp, 2L)
  expect_equal(dim(fit$v), c(4, 2))
  expect_equal(unname(crossprod(fit$v)), diag(2), tolerance = 1e-10)
  expect_equal(
    olda_projection_matrix(fit$v),
    olda_projection_matrix(source$v),
    tolerance = 1e-9
  )
})

test_that("OLDA matches a direct generalized-eigenspace oracle", {
  dat <- olda_fixture()
  fit <- olda(dat$X, dat$y)
  scatter <- olda_scatter_reference(dat$X, dat$y)

  direct <- eigen(solve(scatter$St, scatter$Sb))
  order_direct <- order(Re(direct$values), decreasing = TRUE)
  direct_vectors <- Re(direct$vectors[, order_direct[seq_len(fit$ncomp)], drop = FALSE])
  direct_values <- Re(direct$values[order_direct[seq_len(fit$ncomp)]])

  expect_equal(
    olda_projection_matrix(fit$v),
    olda_projection_matrix(direct_vectors),
    tolerance = 1e-8
  )
  expect_equal(
    unname(fit$ulda_eigenvalues),
    direct_values,
    tolerance = 1e-9
  )

  reduced_total <- crossprod(fit$v, scatter$St %*% fit$v)
  reduced_between <- crossprod(fit$v, scatter$Sb %*% fit$v)
  criterion <- sum(diag(solve(reduced_total, reduced_between)))
  expect_equal(criterion, sum(direct_values), tolerance = 1e-9)
})

test_that("OLDA retains an auditable QR and score transformation", {
  dat <- olda_fixture()
  fit <- olda(dat$X, dat$y)
  source <- ulda(dat$X, dat$y)
  pivot <- fit$ulda_pivot

  expect_equal(
    fit$v %*% fit$qr_R,
    source$v[, pivot, drop = FALSE],
    tolerance = 1e-10
  )
  expect_equal(
    fit$s %*% fit$qr_R,
    source$s[, pivot, drop = FALSE],
    tolerance = 1e-10
  )
  expect_equal(sort(pivot), seq_len(fit$ncomp))
  expect_equal(colnames(fit$v), c("OLDA1", "OLDA2"))
  expect_equal(rownames(fit$v), colnames(dat$X))
})

test_that("OLDA training and new-data projections are coherent", {
  dat <- olda_fixture()
  fit <- olda(dat$X, dat$y, ncomp = 1)

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

  fit_pass <- olda(dat$X, dat$y, preproc = multivarious::pass())
  expect_equal(
    multivarious::project(fit_pass, dat$X),
    fit_pass$s,
    tolerance = 1e-10
  )
})

test_that("OLDA handles rank-one and undersampled problems", {
  X_one <- matrix(c(-3, -2, -1, 1, 2, 4), ncol = 1)
  y_one <- factor(rep(c("left", "right"), each = 3))
  fit_one <- olda(X_one, y_one)

  expect_equal(dim(fit_one$v), c(1, 1))
  expect_equal(unname(crossprod(fit_one$v)), matrix(1), tolerance = 1e-12)
  expect_true(all(is.finite(fit_one$s)))

  set.seed(1802)
  y_high <- factor(rep(1:3, each = 3))
  X_high <- matrix(rnorm(length(y_high) * 20), ncol = 20)
  X_high[y_high == 2, 1:3] <- X_high[y_high == 2, 1:3] + 1
  X_high[y_high == 3, 4:6] <- X_high[y_high == 3, 4:6] - 1
  fit_high <- olda(X_high, y_high)

  expect_equal(dim(fit_high$v), c(20, 2))
  expect_equal(unname(crossprod(fit_high$v)), diag(2), tolerance = 1e-10)
  expect_equal(fit_high$total_rank, nrow(X_high) - 1L)
  expect_true(all(is.finite(fit_high$v)))
})

test_that("OLDA is invariant to replication and equivariant to common scaling", {
  dat <- olda_fixture()
  fit <- olda(dat$X, dat$y)
  fit_replicated <- olda(rbind(dat$X, dat$X), factor(rep(dat$y, 2)))
  scale_factor <- 1e-9
  fit_scaled <- olda(dat$X * scale_factor, dat$y)

  expect_equal(fit_replicated$v, fit$v, tolerance = 1e-8)
  expect_equal(
    fit_replicated$s[seq_len(nrow(dat$X)), , drop = FALSE],
    fit$s,
    tolerance = 1e-8
  )
  expect_equal(fit_scaled$v, fit$v, tolerance = 1e-8)
  expect_equal(fit_scaled$s / scale_factor, fit$s, tolerance = 1e-7)
})

test_that("randomized OLDA fits preserve orthogonality and ULDA subspaces", {
  for (seed in 1:5) {
    set.seed(seed)
    y <- factor(rep(1:3, each = 6))
    X <- matrix(rnorm(length(y) * 6), ncol = 6)
    X[y == 2, 1:2] <- X[y == 2, 1:2] + 0.7
    X[y == 3, 3:4] <- X[y == 3, 3:4] - 0.7

    fit <- olda(X, y, ncomp = 2)
    source <- ulda(X, y, ncomp = 2)

    expect_true(all(is.finite(fit$v)), info = paste("seed", seed))
    expect_true(all(is.finite(fit$s)), info = paste("seed", seed))
    expect_equal(
      unname(crossprod(fit$v)), diag(2), tolerance = 1e-9,
      info = paste("seed", seed)
    )
    expect_equal(
      olda_projection_matrix(fit$v),
      olda_projection_matrix(source$v),
      tolerance = 1e-8,
      info = paste("seed", seed)
    )
  }
})

test_that("OLDA and direct fastolda span the same unregularized subspace", {
  dat <- olda_fixture()
  y_unused <- factor(dat$y, levels = c(levels(dat$y), "unused"))
  fit <- olda(as.data.frame(dat$X), y_unused)
  fit_fast <- fastolda(dat$X, dat$y, reg = 0)

  expect_equal(nlevels(fit$labels), 3L)
  expect_equal(
    olda_projection_matrix(fit_fast$v),
    olda_projection_matrix(fit$v),
    tolerance = 1e-9
  )
  expect_equal(unname(crossprod(fit_fast$v)), diag(2), tolerance = 1e-10)
  expect_equal(
    multivarious::project(fit_fast, dat$X), fit_fast$s,
    tolerance = 1e-10
  )
  expect_identical(fit_fast$algorithm, "direct_svd_qr")
  expect_equal(fit_fast$ridge, 0)

  fit_mu <- olda(dat$X, dat$y, mu = 10)
  expect_equal(fit_mu$v, fit$v, tolerance = 1e-10)
  expect_equal(fit_mu$mu, 10)
})

test_that("fastolda uses its ridge and handles undersampled data directly", {
  dat <- olda_fixture()
  unregularized <- fastolda(dat$X, dat$y, reg = 0)
  regularized <- fastolda(dat$X, dat$y, reg = 0.5)

  expect_gt(regularized$ridge, 0)
  expect_equal(regularized$reg, 0.5)
  expect_false(isTRUE(all.equal(
    olda_projection_matrix(unregularized$v),
    olda_projection_matrix(regularized$v),
    tolerance = 1e-12
  )))

  set.seed(1882)
  y <- factor(rep(1:3, each = 3))
  X <- matrix(rnorm(length(y) * 30), nrow = length(y))
  X[y == 2, 1:4] <- X[y == 2, 1:4] + 1
  X[y == 3, 5:8] <- X[y == 3, 5:8] - 1
  fit <- fastolda(X, y, reg = 0.01)

  expect_equal(dim(fit$v), c(30, 2))
  expect_equal(unname(crossprod(fit$v)), diag(2), tolerance = 1e-10)
  expect_true(all(is.finite(fit$v)))
  expect_equal(fit$total_rank, nrow(X) - 1L)
})

test_that("OLDA delegates validation and rank failures to ULDA", {
  dat <- olda_fixture()
  X <- dat$X
  y <- dat$y

  expect_error(olda(X, y[-1]), "one non-missing label")
  expect_error(olda(X, rep("one", nrow(X))), "at least two classes")
  expect_error(olda(X, y, mu = -1), "mu")
  expect_error(olda(X, y, tol = 0), "tol")
  expect_error(olda(X, y, ncomp = 0), "ncomp")
  expect_error(olda(X, y, ncomp = 1.5), "integer")

  X_bad <- X
  X_bad[1, 1] <- Inf
  expect_error(olda(X_bad, y), "finite")

  labels <- factor(rep(c("a", "b"), each = 3))
  expect_error(
    olda(matrix(1, nrow = 6, ncol = 2), labels),
    "zero numerical total-scatter rank"
  )
})
