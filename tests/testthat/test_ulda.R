ulda_fixture <- function() {
  set.seed(1701)
  y <- factor(c(rep("a", 5), rep("b", 6), rep("c", 7)))
  means <- rbind(
    c(-1.5, 0.5, 0, 0),
    c(1, -0.5, 0.5, 0),
    c(0, 1.5, -1, 0.5)
  )
  X <- matrix(rnorm(length(y) * 4, sd = 0.45), ncol = 4)
  X <- X + means[as.integer(y), ]
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  list(X = X, y = y)
}

ulda_scatter_reference <- function(X, y) {
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

orthogonal_projection <- function(A) {
  Q <- qr.Q(qr(A))
  tcrossprod(Q)
}

test_that("ULDA satisfies its defining scatter identities", {
  dat <- ulda_fixture()
  fit <- ulda(dat$X, dat$y)
  scatter <- ulda_scatter_reference(dat$X, dat$y)
  Sw <- scatter$St - scatter$Sb

  expect_s3_class(fit, "ulda")
  expect_s3_class(fit, "discriminant_projector")
  expect_equal(fit$ncomp, 2L)
  expect_equal(dim(fit$v), c(4, 2))
  expect_equal(
    unname(crossprod(fit$v, scatter$St %*% fit$v)),
    diag(2),
    tolerance = 1e-9
  )
  expect_equal(
    unname(crossprod(fit$v, scatter$Sb %*% fit$v)),
    diag(fit$eigenvalues),
    tolerance = 1e-9
  )
  expect_equal(
    unname(crossprod(fit$v, Sw %*% fit$v)),
    diag(1 - fit$eigenvalues),
    tolerance = 1e-9
  )

  generalized_residual <- scatter$Sb %*% fit$v -
    scatter$St %*% fit$v %*% diag(fit$eigenvalues)
  expect_lt(max(abs(generalized_residual)), 1e-9)
})

test_that("ULDA matches a direct generalized-eigenvalue oracle", {
  dat <- ulda_fixture()
  fit <- ulda(dat$X, dat$y)
  scatter <- ulda_scatter_reference(dat$X, dat$y)

  direct <- eigen(solve(scatter$St, scatter$Sb))
  order_direct <- order(Re(direct$values), decreasing = TRUE)
  direct_values <- Re(direct$values[order_direct[seq_len(fit$ncomp)]])
  direct_vectors <- Re(direct$vectors[, order_direct[seq_len(fit$ncomp)], drop = FALSE])

  expect_equal(fit$eigenvalues, direct_values, tolerance = 1e-9)
  expect_equal(
    orthogonal_projection(fit$v),
    orthogonal_projection(direct_vectors),
    tolerance = 1e-8
  )
})

test_that("ULDA scores are uncorrelated and projector-compatible", {
  dat <- ulda_fixture()
  fit <- ulda(dat$X, dat$y, ncomp = 2)
  centered_scores <- sweep(fit$s, 2, colMeans(fit$s), "-")

  expect_equal(
    unname(crossprod(centered_scores) / nrow(dat$X)),
    diag(2),
    tolerance = 1e-9
  )
  expect_equal(
    unname(fit$sdev),
    rep(sqrt(nrow(dat$X) / (nrow(dat$X) - 1)), 2),
    tolerance = 1e-9
  )
  expect_equal(
    multivarious::project(fit, dat$X),
    fit$s,
    tolerance = 1e-10
  )
  expect_equal(
    dim(multivarious::project(fit, dat$X[1:3, , drop = FALSE])),
    c(3, 2)
  )
  expect_equal(colnames(fit$v), c("ULDA1", "ULDA2"))
  expect_equal(rownames(fit$v), colnames(dat$X))

  pivots <- vapply(seq_len(ncol(fit$v)), function(j) {
    fit$v[which.max(abs(fit$v[, j])), j]
  }, numeric(1))
  expect_true(all(pivots > 0))
})

test_that("ULDA handles rank-one and undersampled problems", {
  X_one <- matrix(c(-3, -2, -1, 1, 2, 4), ncol = 1)
  y_one <- factor(rep(c("left", "right"), each = 3))
  fit_one <- ulda(X_one, y_one)

  expect_equal(dim(fit_one$v), c(1, 1))
  expect_true(all(is.finite(fit_one$v)))
  expect_equal(fit_one$ncomp, 1L)

  set.seed(1702)
  y_high <- factor(rep(1:3, each = 3))
  X_high <- matrix(rnorm(length(y_high) * 20), ncol = 20)
  X_high[y_high == 2, 1:3] <- X_high[y_high == 2, 1:3] + 1
  X_high[y_high == 3, 4:6] <- X_high[y_high == 3, 4:6] - 1
  fit_high <- ulda(X_high, y_high)

  expect_equal(dim(fit_high$v), c(20, 2))
  expect_equal(fit_high$total_rank, nrow(X_high) - 1L)
  expect_true(all(is.finite(fit_high$v)))
  expect_true(all(is.finite(fit_high$s)))
})

test_that("ULDA is invariant to replication and common predictor scaling", {
  dat <- ulda_fixture()
  fit <- ulda(dat$X, dat$y)
  fit_replicated <- ulda(rbind(dat$X, dat$X), factor(rep(dat$y, 2)))
  scale_factor <- 1e-9
  fit_scaled <- ulda(dat$X * scale_factor, dat$y)

  expect_equal(fit_replicated$v, fit$v, tolerance = 1e-8)
  expect_equal(fit_replicated$eigenvalues, fit$eigenvalues, tolerance = 1e-10)
  expect_equal(
    fit_scaled$v * scale_factor,
    fit$v,
    tolerance = 1e-7
  )
  expect_equal(fit_scaled$s, fit$s, tolerance = 1e-8)
  expect_equal(fit_scaled$eigenvalues, fit$eigenvalues, tolerance = 1e-9)
})

test_that("ULDA drops unused classes and honors component selection", {
  dat <- ulda_fixture()
  y_unused <- factor(dat$y, levels = c(levels(dat$y), "unused"))
  fit <- ulda(as.data.frame(dat$X), y_unused, ncomp = 1)

  expect_equal(fit$ncomp, 1L)
  expect_equal(nlevels(fit$labels), 3L)
  expect_equal(dim(fit$v), c(4, 1))

  fit_mu <- ulda(dat$X, dat$y, ncomp = 1, mu = 10)
  expect_equal(fit_mu$v, fit$v, tolerance = 1e-10)
  expect_equal(fit_mu$mu, 10)
})

test_that("ULDA remains compatible with the downstream OLDA wrapper", {
  dat <- ulda_fixture()
  fit <- olda(dat$X, dat$y)

  expect_s3_class(fit, "olda")
  expect_equal(dim(fit$v), c(4, 2))
  expect_equal(
    unname(crossprod(fit$v)),
    diag(2),
    tolerance = 1e-10
  )
  expect_equal(
    multivarious::project(fit, dat$X),
    fit$s,
    tolerance = 1e-10
  )
})

test_that("ULDA reports degenerate scatter and effective-rank failures", {
  y <- factor(rep(c("a", "b"), each = 3))
  expect_error(
    ulda(matrix(1, nrow = 6, ncol = 2), y),
    "zero numerical total-scatter rank"
  )

  shared_cloud <- rbind(c(-1, 0), c(0, 1), c(1, -1))
  expect_error(
    ulda(rbind(shared_cloud, shared_cloud), y),
    "zero numerical between-class scatter"
  )

  X_rank_one_between <- rbind(
    c(-2, -1), c(-2, 1),
    c(0, -1), c(0, 1),
    c(2, -1), c(2, 1)
  )
  y_three <- factor(rep(c("a", "b", "c"), each = 2))
  expect_error(
    ulda(X_rank_one_between, y_three, ncomp = 2),
    "exceeds the numerical between-class rank"
  )
})

test_that("ULDA validates data, labels, and numerical controls", {
  dat <- ulda_fixture()
  X <- dat$X
  y <- dat$y

  expect_error(ulda(matrix(letters[1:8], nrow = 4), rep(1:2, each = 2)), "numeric matrix")
  expect_error(ulda(X, y[-1]), "one non-missing label")
  y_missing <- y
  y_missing[1] <- NA
  expect_error(ulda(X, y_missing), "one non-missing label")
  expect_error(ulda(X, rep("one", nrow(X))), "at least two classes")
  expect_error(ulda(X, y, mu = -1), "mu")
  expect_error(ulda(X, y, tol = 0), "tol")
  expect_error(ulda(X, y, tol = 1), "tol")
  expect_error(ulda(X, y, ncomp = 0), "ncomp")
  expect_error(ulda(X, y, ncomp = 1.5), "integer")

  X_bad <- X
  X_bad[1, 1] <- Inf
  expect_error(ulda(X_bad, y), "finite")
})
