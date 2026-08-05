gmmsd_test_fixture <- function() {
  set.seed(812)
  y <- factor(rep(letters[1:3], each = 8))
  class_means <- rbind(
    c(-3, 0, 2, 0, 1),
    c(3, 1, -2, 0, -1),
    c(0, -3, 0, 2, 1)
  )
  X <- matrix(rnorm(24 * 5, sd = 0.35), 24, 5) +
    class_means[as.integer(y), ]
  colnames(X) <- paste0("feature", seq_len(ncol(X)))
  list(X = X, y = y)
}

gmmsd_scatter_reference <- function(X, y) {
  y <- droplevels(as.factor(y))
  classes <- levels(y)
  overall <- colMeans(X)
  within <- matrix(0, ncol(X), ncol(X))
  between <- matrix(0, ncol(X), ncol(X))

  for (class_name in classes) {
    rows <- which(y == class_name)
    class_mean <- colMeans(X[rows, , drop = FALSE])
    for (row in rows) {
      within <- within + tcrossprod(X[row, ] - class_mean)
    }
    between <- between + length(rows) * tcrossprod(class_mean - overall)
  }
  list(within = within, between = between)
}

gmmsd_subspace_projector <- function(vectors) {
  vectors %*% t(vectors)
}

test_that("GMMSD solves the defining scatter-difference eigenproblem", {
  dat <- gmmsd_test_fixture()
  balance <- 0.2
  fit <- gmmsd(dat$X, dat$y, c = balance, dim = 2)
  Xp <- multivarious::transform(fit$preproc, dat$X)
  scatter <- gmmsd_scatter_reference(Xp, dat$y)
  criterion <- scatter$between - balance * scatter$within
  residual <- criterion %*% fit$v -
    fit$v %*% diag(fit$eigenvalues, nrow = fit$dim)

  expect_s3_class(fit, "gmmsd")
  expect_equal(unname(crossprod(fit$v)), diag(fit$dim), tolerance = 1e-10)
  expect_lt(norm(residual, type = "F"), 1e-8 * norm(criterion, type = "F"))
  expect_equal(
    unname(crossprod(fit$v, criterion %*% fit$v)),
    diag(fit$eigenvalues, nrow = fit$dim),
    tolerance = 1e-9
  )
  expect_equal(fit$s, unname(Xp %*% fit$v), tolerance = 1e-12,
               ignore_attr = "dimnames")
  expect_equal(rownames(fit$v), colnames(dat$X))
  expect_equal(colnames(fit$v), paste0("GMMSD", seq_len(fit$dim)))
  expect_equal(fit$c, balance)
  expect_equal(fit$dim, 2L)
  expect_true(all(is.finite(fit$sdev)))
})

test_that("GMMSD agrees with the direct full-space oracle", {
  dat <- gmmsd_test_fixture()
  balance <- 0.5
  fit <- gmmsd(dat$X, dat$y, c = balance, dim = 2)
  Xp <- multivarious::transform(fit$preproc, dat$X)
  scatter <- gmmsd_scatter_reference(Xp, dat$y)
  direct <- eigen(scatter$between - balance * scatter$within, symmetric = TRUE)
  direct_vectors <- direct$vectors[, 1:2, drop = FALSE]

  expect_equal(unname(fit$eigenvalues), direct$values[1:2], tolerance = 1e-10)
  expect_equal(
    unname(gmmsd_subspace_projector(fit$v)),
    unname(gmmsd_subspace_projector(direct_vectors)),
    tolerance = 1e-9
  )
})

test_that("GMMSD is equivariant to rows, translation, and global scale", {
  dat <- gmmsd_test_fixture()
  baseline <- gmmsd(dat$X, dat$y, c = 0.3, dim = 2,
                     preproc = multivarious::pass())
  shift <- c(100, -500, 250, 75, -30)
  shifted_X <- sweep(dat$X, 2, shift, "+")
  shifted <- gmmsd(shifted_X, dat$y, c = 0.3, dim = 2,
                   preproc = multivarious::pass())
  permutation <- c(24:13, 1:12)
  permuted <- gmmsd(dat$X[permutation, ], dat$y[permutation],
                    c = 0.3, dim = 2, preproc = multivarious::pass())

  expect_equal(
    gmmsd_subspace_projector(shifted$v),
    gmmsd_subspace_projector(baseline$v),
    tolerance = 1e-9
  )
  expect_equal(
    unname(sweep(shifted$s, 2, colMeans(shifted$s), "-")),
    unname(sweep(baseline$s, 2, colMeans(baseline$s), "-")),
    tolerance = 1e-9
  )
  for (scale_factor in c(1e-8, 1e6, 1e8)) {
    scaled <- gmmsd(scale_factor * dat$X, dat$y, c = 0.3, dim = 2,
                    preproc = multivarious::pass())
    expect_equal(
      gmmsd_subspace_projector(scaled$v),
      gmmsd_subspace_projector(baseline$v),
      tolerance = 1e-9
    )
    expect_equal(
      unname(scaled$eigenvalues),
      scale_factor^2 * unname(baseline$eigenvalues),
      tolerance = 1e-8
    )
  }
  expect_equal(
    gmmsd_subspace_projector(permuted$v),
    gmmsd_subspace_projector(baseline$v),
    tolerance = 1e-9
  )
})

test_that("GMMSD restricts high-dimensional fits to numerical row rank", {
  set.seed(913)
  latent <- matrix(rnorm(12 * 4), 12, 4)
  loadings <- matrix(rnorm(4 * 50), 4, 50)
  X <- latent %*% loadings
  y <- factor(rep(letters[1:3], each = 4))
  fit <- gmmsd(X, y, c = 1, dim = 4, tol = 1e-8)

  centered <- sweep(X, 2, colMeans(X), "-")
  row_basis <- svd(centered, nu = 0, nv = 4)$v
  row_projector <- row_basis %*% t(row_basis)

  expect_equal(fit$optimization_rank, 4L)
  expect_equal(dim(fit$v), c(50L, 4L))
  expect_equal(unname(crossprod(fit$v)), diag(4), tolerance = 1e-9)
  expect_lt(norm(fit$v - row_projector %*% fit$v, type = "F"), 1e-8)
  expect_error(gmmsd(X, y, dim = 5, tol = 1e-8), "numerical rank")
})

test_that("GMMSD reports deterministic numerical diagnostics", {
  dat <- gmmsd_test_fixture()
  fit1 <- gmmsd(dat$X, dat$y, c = 0.4, dim = 2, tol = 1e-7)
  fit2 <- gmmsd(dat$X, dat$y, c = 0.4, dim = 2, tol = 1e-7)
  fit_alias <- gmmsd(dat$X, dat$y, c = 0.4, ncomp = 2, tol = 1e-7)

  expect_equal(fit1$v, fit2$v, tolerance = 0)
  expect_equal(fit1$v, fit_alias$v, tolerance = 0)
  expect_equal(fit1$s, fit2$s, tolerance = 0)
  expect_equal(fit1$optimization_rank, 5L)
  expect_length(fit1$qr_diagonal, min(dim(dat$X)))
  expect_equal(fit1$tol, 1e-7)
  expect_true(all(diff(fit1$eigenvalues) <= 0))
  expect_length(fit1$criterion_eigenvalues, fit1$optimization_rank)
  expect_true(is.finite(fit1$data_scale) && fit1$data_scale > 0)
  expect_true(
    is.finite(fit1$criterion_scale_normalized) &&
      fit1$criterion_scale_normalized > 0
  )
  expect_length(fit1$criterion_center, ncol(dat$X))
  expect_no_error(gmmsd(dat$X, dat$y, c = 0, dim = 2))
  expect_error(
    gmmsd(dat$X, dat$y, dim = 1, ncomp = 2),
    "compatibility alias"
  )
})

test_that("GMMSD rejects invalid controls, labels, and transformed data", {
  dat <- gmmsd_test_fixture()

  expect_error(gmmsd(matrix(letters[1:24], 8, 3), rep(1:2, each = 4)),
               "numeric matrix")
  contaminated <- dat$X
  contaminated[1, 1] <- Inf
  expect_error(gmmsd(contaminated, dat$y), "finite")
  expect_error(gmmsd(dat$X, dat$y[-1]), "one non-missing label")
  expect_error(gmmsd(dat$X, replace(dat$y, 1, NA)), "one non-missing label")
  expect_error(gmmsd(dat$X, rep("one", nrow(dat$X))), "at least two classes")

  for (bad_c in list(-1, Inf, NA_real_, c(1, 2))) {
    expect_error(gmmsd(dat$X, dat$y, c = bad_c), "'c'")
  }
  for (bad_dim in list(0, 1.5, Inf, NA_real_, c(1, 2))) {
    expect_error(gmmsd(dat$X, dat$y, dim = bad_dim), "'dim'")
  }
  for (bad_tol in list(0, 1, -1, Inf, NA_real_, c(1e-8, 1e-6))) {
    expect_error(gmmsd(dat$X, dat$y, tol = bad_tol), "'tol'")
  }

  expect_error(
    gmmsd(matrix(1, nrow = 6, ncol = 3), rep(letters[1:2], each = 3), dim = 1),
    "zero numerical rank"
  )
})
