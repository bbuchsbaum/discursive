soft_lda_projection_matrix <- function(v) {
  q <- qr.Q(qr(v))
  tcrossprod(q)
}

soft_lda_fixture <- function() {
  X <- rbind(
    c(-2.0, 0.2, 1.0), c(-1.2, -0.3, 0.7), c(-0.8, 0.4, 1.2),
    c(1.0, 1.8, -0.2), c(1.5, 1.1, 0.1), c(2.1, 1.4, -0.5),
    c(0.0, -1.8, -1.0), c(0.5, -1.2, -1.4), c(-0.4, -1.4, -0.7)
  )
  C <- rbind(
    c(1.0, 0.1, 0.0), c(0.8, 0.2, 0.0), c(0.9, 0.1, 0.1),
    c(0.1, 1.0, 0.0), c(0.0, 0.9, 0.1), c(0.2, 0.8, 0.0),
    c(0.0, 0.1, 1.0), c(0.1, 0.0, 0.9), c(0.1, 0.2, 0.8)
  )
  colnames(X) <- paste0("x", 1:3)
  colnames(C) <- c("a", "b", "c")
  list(X = X, C = C)
}

em_test_log_normalize <- function(log_weights) {
  row_max <- apply(log_weights, 1L, max)
  scaled <- exp(sweep(log_weights, 1L, row_max, FUN = "-"))
  totals <- rowSums(scaled)
  list(
    weights = sweep(scaled, 1L, totals, FUN = "/"),
    log_norm = row_max + log(totals)
  )
}

em_test_lda_state <- function(fit, X, PL) {
  n <- nrow(X)
  d <- ncol(X)
  K <- ncol(PL)
  covariance_factor <- chol(as.matrix(fit$Sigma))
  logdet <- 2 * sum(log(diag(covariance_factor)))
  log_density <- vapply(seq_len(K), function(k) {
    difference <- sweep(X, 2L, fit$mu[k, ], FUN = "-")
    whitened <- forwardsolve(t(covariance_factor), t(difference))
    -0.5 * (d * log(2 * pi) + logdet + colSums(whitened^2))
  }, numeric(n))
  log_joint <- log(PL) + sweep(
    log_density, 2L, log(as.numeric(fit$pi)), FUN = "+"
  )
  normalized <- em_test_log_normalize(log_joint)
  list(zeta = normalized$weights, loglik = sum(normalized$log_norm))
}

em_test_lr_state <- function(beta, X, PL) {
  eta <- cbind(1, X) %*% beta
  augmented <- cbind(eta, 0)
  row_max <- apply(augmented, 1L, max)
  shifted <- exp(sweep(augmented, 1L, row_max, FUN = "-"))
  probability <- sweep(shifted, 1L, rowSums(shifted), FUN = "/")
  log_probability <- log(probability)
  normalized <- em_test_log_normalize(log(PL) + log_probability)
  list(
    probability = probability,
    zeta = normalized$weights,
    loglik = sum(normalized$log_norm)
  )
}

em_test_lr_objective <- function(beta, X, PL, lambda) {
  state <- em_test_lr_state(beta, X, PL)
  state$loglik - 0.5 * lambda * sum(beta^2)
}

em_test_crisp_labels <- function(y, K = max(y)) {
  base::diag(K)[y, , drop = FALSE]
}

test_that("the projection and evidential-EM soft-label APIs are distinct", {
  expect_true(is.function(soft_lda))
  expect_true(is.function(em_soft_lda))
  expect_true("C" %in% names(formals(soft_lda)))
  expect_true("PL" %in% names(formals(em_soft_lda)))

  dat <- soft_lda_fixture()
  set.seed(401)
  fit <- em_soft_lda(dat$X, dat$C, n_starts = 1, max_iter = 20)
  expect_named(
    fit,
    c("pi", "mu", "Sigma", "zeta", "loglik", "iter", "diagnostics")
  )
  expect_equal(rowSums(fit$zeta), rep(1, nrow(dat$X)), tolerance = 1e-10)
  expect_true(all(is.finite(fit$Sigma)))
})

test_that("SL-LDA scatter matrices match the dense E/F/G definitions", {
  dat <- soft_lda_fixture()
  alpha <- 0.2
  fit <- soft_lda(dat$X, dat$C, dp = 3, di = 3, dl = 2, alpha = alpha)

  Xc <- sweep(dat$X, 2L, fit$pca_center, FUN = "-")
  Z <- Xc %*% fit$pca_vectors
  F <- t(dat$C)
  E <- diag(rowSums(dat$C))
  Ginv <- diag(1 / colSums(dat$C))
  within_operator <- E - t(F) %*% Ginv %*% F
  between_operator <- t(F) %*% Ginv %*% F -
    E %*% matrix(1, nrow(dat$X), nrow(dat$X)) %*% E / sum(E)
  Sw <- crossprod(Z, within_operator %*% Z)
  Sb <- crossprod(Z, between_operator %*% Z)

  expect_equal(fit$within_scatter, Sw, tolerance = 1e-10)
  expect_equal(fit$between_scatter, Sb, tolerance = 1e-10)

  W <- crossprod(fit$pca_vectors, fit$v)
  for (j in seq_len(ncol(W))) {
    residual <- Sb %*% W[, j] -
      fit$eigenvalues[j] * (Sw + alpha * diag(3)) %*% W[, j]
    expect_lt(sqrt(sum(residual^2)), 1e-8)
  }
})

test_that("SL-LDA selects the largest positive within-class modes", {
  dat <- soft_lda_fixture()
  fit <- soft_lda(dat$X, dat$C, dp = 3, di = 2, dl = 2, alpha = 0.1)
  expected <- eigen(
    fit$within_scatter + 0.1 * diag(3), symmetric = TRUE,
    only.values = TRUE
  )$values[1:2]

  expect_equal(fit$within_eigenvalues, expected, tolerance = 1e-12)
  expect_true(all(diff(fit$within_eigenvalues) <= 0))
  expect_equal(multivarious::project(fit, dat$X), fit$s, tolerance = 1e-12)
  expect_identical(levels(fit$labels), colnames(dat$C))
})

test_that("SL-LDA is invariant to common membership scaling and translation", {
  dat <- soft_lda_fixture()
  fit <- soft_lda(dat$X, dat$C, dp = 3, di = 3, dl = 2, alpha = 0)
  scaled <- soft_lda(dat$X, 7 * dat$C, dp = 3, di = 3, dl = 2, alpha = 0)
  shift <- c(100, -50, 25)
  shifted_X <- sweep(dat$X, 2L, shift, FUN = "+")
  shifted <- soft_lda(shifted_X, dat$C, dp = 3, di = 3, dl = 2, alpha = 0)

  expect_equal(
    soft_lda_projection_matrix(fit$v),
    soft_lda_projection_matrix(scaled$v), tolerance = 1e-9
  )
  expect_equal(fit$eigenvalues, scaled$eigenvalues, tolerance = 1e-10)
  expect_equal(
    soft_lda_projection_matrix(fit$v),
    soft_lda_projection_matrix(shifted$v), tolerance = 1e-9
  )
  expect_equal(as.matrix(dist(fit$s)), as.matrix(dist(shifted$s)), tolerance = 1e-9)
})

test_that("SL-LDA does not use RNG and validates membership contracts", {
  dat <- soft_lda_fixture()
  set.seed(902)
  before <- .Random.seed
  invisible(soft_lda(dat$X, dat$C, dl = 2))
  expect_identical(.Random.seed, before)

  C <- dat$C
  C[1, ] <- 0
  expect_error(soft_lda(dat$X, C), "Every row")
  C <- dat$C
  C[, 3] <- 0
  expect_error(soft_lda(dat$X, C), "Every column")
  C <- dat$C
  C[1, 1] <- -1
  expect_error(soft_lda(dat$X, C), "nonnegative")
  expect_error(soft_lda(dat$X, dat$C, dp = 1.5), "integer")
  expect_error(soft_lda(dat$X, dat$C, tol = 0), "tol")
})

test_that("evidential LDA matches the crisp-label maximum-likelihood oracle", {
  dat <- soft_lda_fixture()
  y <- rep(seq_len(3), each = 3)
  PL <- em_test_crisp_labels(y, 3)
  reg <- 1e-6

  set.seed(1201)
  fit <- em_soft_lda(
    dat$X, PL, max_iter = 1, tol = 1e-12, n_starts = 1, reg = reg
  )

  counts <- tabulate(y, nbins = 3)
  expected_mu <- t(vapply(seq_len(3), function(k) {
    colMeans(dat$X[y == k, , drop = FALSE])
  }, numeric(ncol(dat$X))))
  expected_scatter <- matrix(0, ncol(dat$X), ncol(dat$X))
  for (k in seq_len(3)) {
    difference <- sweep(
      dat$X[y == k, , drop = FALSE], 2L, expected_mu[k, ], FUN = "-"
    )
    expected_scatter <- expected_scatter + crossprod(difference)
  }
  expected_covariance <- expected_scatter / nrow(dat$X) +
    base::diag(reg, ncol(dat$X))
  reference <- em_test_lda_state(fit, dat$X, PL)

  expect_equal(as.numeric(fit$pi), counts / length(y), tolerance = 1e-12)
  expect_equal(unname(fit$mu), unname(expected_mu), tolerance = 1e-12)
  expect_equal(fit$Sigma, expected_covariance, tolerance = 1e-11)
  expect_equal(unname(fit$zeta), PL, tolerance = 0)
  expect_equal(fit$loglik, reference$loglik, tolerance = 1e-10)
  expect_identical(fit$iter, 1L)
})

test_that("constrained LDA covariance policies match crisp-label oracles", {
  dat <- soft_lda_fixture()
  y <- rep(seq_len(3), each = 3)
  PL <- em_test_crisp_labels(y, 3)
  reg <- 2e-6
  expected_mu <- t(vapply(seq_len(3), function(k) {
    colMeans(dat$X[y == k, , drop = FALSE])
  }, numeric(ncol(dat$X))))
  residual <- dat$X - expected_mu[y, , drop = FALSE]
  expected_variance <- colSums(residual^2) / nrow(dat$X) + reg
  expected_spherical <- mean(colSums(residual^2) / nrow(dat$X)) + reg

  for (policy in c("diagonal", "spherical")) {
    set.seed(1211)
    fit <- em_soft_lda(
      dat$X, PL, max_iter = 1, tol = 1e-12,
      n_starts = 1, reg = reg, covariance = policy
    )
    expected_diagonal <- if (policy == "diagonal") {
      expected_variance
    } else {
      rep(expected_spherical, ncol(dat$X))
    }
    reference <- em_test_lda_state(fit, dat$X, PL)

    expect_s4_class(fit$Sigma, "diagonalMatrix")
    expect_equal(
      unname(diag(as.matrix(fit$Sigma))),
      unname(expected_diagonal), tolerance = 1e-12
    )
    expect_equal(
      as.matrix(fit$Sigma) - diag(diag(as.matrix(fit$Sigma))),
      matrix(0, ncol(dat$X), ncol(dat$X)), tolerance = 0
    )
    expect_equal(unname(fit$mu), unname(expected_mu), tolerance = 1e-12)
    expect_equal(unname(fit$zeta), PL, tolerance = 0)
    expect_equal(fit$loglik, reference$loglik, tolerance = 1e-10)
    expect_identical(fit$diagnostics$covariance$policy, policy)
  }
})

test_that("evidential LDA exposes convergence, start, and covariance diagnostics", {
  dat <- soft_lda_fixture()
  set.seed(1212)
  fit <- em_soft_lda(
    dat$X, dat$C, max_iter = 400, tol = 1e-10,
    n_starts = 3, reg = 1e-7, covariance = "diagonal"
  )
  diagnostics <- fit$diagnostics
  chosen <- diagnostics$starts[
    diagnostics$starts$start == diagnostics$chosen_start, , drop = FALSE
  ]

  expect_true(diagnostics$converged)
  expect_identical(
    diagnostics$termination_reason, "relative_loglik_tolerance"
  )
  expect_equal(tail(diagnostics$loglik_history, 1), fit$loglik)
  expect_length(diagnostics$loglik_history, fit$iter + 1L)
  expect_true(all(diff(diagnostics$loglik_history) >= -1e-10))
  expect_equal(diagnostics$final_loglik_change, tail(
    diff(diagnostics$loglik_history), 1
  ))
  expect_equal(nrow(diagnostics$starts), 3L)
  expect_true(all(diagnostics$starts$status == "completed"))
  expect_equal(chosen$loglik, max(diagnostics$starts$loglik))
  expect_identical(chosen$converged, diagnostics$converged)
  expect_identical(diagnostics$covariance$policy, "diagonal")
  expect_identical(diagnostics$covariance$storage, "sparse_diagonal")
  expect_equal(diagnostics$covariance$requested_ridge, 1e-7)
  expect_true(all(unlist(diagnostics$covariance[3:5]) >= 0))
  expect_equal(
    diagnostics$class_mapping,
    data.frame(pl_column = 1:3, class = c("a", "b", "c"))
  )

  set.seed(1213)
  limited <- em_soft_lda(
    dat$X, dat$C, max_iter = 1, tol = 1e-15,
    n_starts = 1, covariance = "spherical"
  )
  expect_false(limited$diagnostics$converged)
  expect_identical(limited$diagnostics$termination_reason, "max_iter")
})

test_that("a failed LDA start is recorded while later starts continue", {
  dat <- soft_lda_fixture()
  original_runner <- discursive:::.disc_run_lda_start
  testthat::local_mocked_bindings(
    .disc_run_lda_start = function(start_id, ...) {
      if (start_id == 1L) {
        condition <- structure(
          list(
            message = "deliberate initialization failure",
            call = NULL,
            reason = "initialization_failed"
          ),
          class = c("disc_lda_start_error", "error", "condition")
        )
        stop(condition)
      }
      original_runner(start_id = start_id, ...)
    },
    .package = "discursive"
  )

  set.seed(1214)
  fit <- em_soft_lda(
    dat$X, dat$C, max_iter = 20, n_starts = 2,
    covariance = "spherical"
  )

  expect_identical(fit$diagnostics$chosen_start, 2L)
  expect_identical(fit$diagnostics$starts$status, c("failed", "completed"))
  expect_identical(
    fit$diagnostics$starts$termination_reason[1], "initialization_failed"
  )
  expect_match(fit$diagnostics$starts$message[1], "deliberate")
  expect_true(is.finite(fit$loglik))
})

test_that("evidential LDA returns a self-consistent converged EM state", {
  dat <- soft_lda_fixture()
  reg <- 1e-7
  d <- ncol(dat$X)
  n <- nrow(dat$X)

  for (policy in c("full", "diagonal", "spherical")) {
    set.seed(1202)
    fit <- em_soft_lda(
      dat$X, dat$C, max_iter = 400, tol = 1e-12,
      n_starts = 1, reg = reg, covariance = policy
    )
    reference <- em_test_lda_state(fit, dat$X, dat$C)
    nk <- colSums(fit$zeta)
    fixed_mu <- crossprod(fit$zeta, dat$X) / nk
    fixed_scatter <- matrix(0, d, d)
    for (k in seq_len(ncol(dat$C))) {
      difference <- sweep(dat$X, 2L, fixed_mu[k, ], FUN = "-")
      fixed_scatter <- fixed_scatter +
        crossprod(difference, difference * fit$zeta[, k])
    }
    expected_covariance <- switch(
      policy,
      full = fixed_scatter / n + base::diag(reg, d),
      diagonal = base::diag(base::diag(fixed_scatter) / n + reg, d),
      spherical = base::diag(sum(base::diag(fixed_scatter)) / (n * d) + reg, d)
    )

    expect_equal(unname(fit$zeta), unname(reference$zeta), tolerance = 1e-10)
    expect_equal(fit$loglik, reference$loglik, tolerance = 1e-9)
    expect_equal(
      as.numeric(fit$pi), as.numeric(nk / n), tolerance = 2e-6
    )
    expect_equal(unname(fit$mu), unname(fixed_mu), tolerance = 2e-6)
    expect_equal(
      as.matrix(fit$Sigma), expected_covariance, tolerance = 2e-6
    )
    expect_equal(rowSums(fit$zeta), rep(1, n), tolerance = 1e-12)
    expect_true(all(eigen(as.matrix(fit$Sigma), symmetric = TRUE)$values > 0))
  }
})

test_that("each LDA covariance policy is monotone across EM iterations", {
  dat <- soft_lda_fixture()
  for (policy in c("full", "diagonal", "spherical")) {
    likelihood <- vapply(seq_len(10), function(iterations) {
      set.seed(1203)
      em_soft_lda(
        dat$X, dat$C, max_iter = iterations, tol = 1e-14,
        n_starts = 1, reg = 0, covariance = policy
      )$loglik
    }, numeric(1))

    expect_true(
      all(diff(likelihood) >= -1e-9),
      info = paste("non-monotone", policy, "covariance likelihood")
    )
    expect_gt(likelihood[length(likelihood)], likelihood[1])
  }
})

test_that("evidential LDA is invariant to plausibility scale and translation", {
  dat <- soft_lda_fixture()
  row_scale <- exp(seq(-500, 500, length.out = nrow(dat$X)))
  scaled_PL <- sweep(dat$C, 1L, row_scale, FUN = "*")
  shift <- c(1e6, -2e6, 5e5)
  shifted_X <- sweep(dat$X, 2L, shift, FUN = "+")

  for (policy in c("full", "diagonal", "spherical")) {
    set.seed(1204)
    baseline <- em_soft_lda(
      dat$X, dat$C, max_iter = 200, tol = 1e-11,
      n_starts = 1, reg = 1e-7, covariance = policy
    )
    set.seed(1204)
    scaled <- em_soft_lda(
      dat$X, scaled_PL, max_iter = 200, tol = 1e-11,
      n_starts = 1, reg = 1e-7, covariance = policy
    )
    set.seed(1204)
    translated <- em_soft_lda(
      shifted_X, dat$C, max_iter = 200, tol = 1e-11,
      n_starts = 1, reg = 1e-7, covariance = policy
    )

    expect_equal(scaled$pi, baseline$pi, tolerance = 1e-9)
    expect_equal(scaled$mu, baseline$mu, tolerance = 1e-9)
    expect_equal(scaled$Sigma, baseline$Sigma, tolerance = 1e-9)
    expect_equal(scaled$zeta, baseline$zeta, tolerance = 1e-9)
    expect_equal(
      scaled$loglik - baseline$loglik,
      sum(log(row_scale)), tolerance = 1e-8
    )
    expect_equal(translated$pi, baseline$pi, tolerance = 1e-8)
    expect_equal(
      sweep(translated$mu, 2L, shift, FUN = "-"),
      baseline$mu, tolerance = 1e-7
    )
    expect_equal(translated$Sigma, baseline$Sigma, tolerance = 1e-7)
    expect_equal(translated$zeta, baseline$zeta, tolerance = 1e-7)
    expect_equal(translated$loglik, baseline$loglik, tolerance = 1e-7)
  }
})

test_that("evidential LDA handles high-dimensional singular covariance in log space", {
  set.seed(1205)
  X <- matrix(rnorm(12 * 80), 12, 80)
  PL <- matrix(0.2, 12, 3)
  PL[cbind(seq_len(12), rep(seq_len(3), each = 4))] <- 1
  PL <- PL * 1e-300

  set.seed(1206)
  fit <- em_soft_lda(
    X, PL, max_iter = 4, tol = 1e-10, n_starts = 1, reg = 0
  )
  reference <- em_test_lda_state(fit, X, PL)

  expect_true(all(is.finite(c(
    fit$pi, fit$mu, as.matrix(fit$Sigma), fit$zeta,
    fit$loglik, fit$iter
  ))))
  expect_silent(chol(fit$Sigma))
  expect_equal(fit$loglik, reference$loglik, tolerance = 1e-7)
  expect_equal(unname(fit$zeta), reference$zeta, tolerance = 1e-9)
  expect_equal(fit$diagnostics$covariance$requested_ridge, 0)
  expect_gt(fit$diagnostics$covariance$initial_jitter, 0)
  expect_gt(fit$diagnostics$covariance$final_jitter, 0)
  expect_gte(
    fit$diagnostics$covariance$max_jitter,
    fit$diagnostics$covariance$final_jitter
  )
})

test_that("diagonal LDA keeps covariance storage linear in features", {
  set.seed(1207)
  n <- 15L
  d <- 600L
  X <- matrix(rnorm(n * d), n, d)
  PL <- matrix(0.2, n, 3)
  PL[cbind(seq_len(n), rep(seq_len(3), each = 5))] <- 1
  fit <- em_soft_lda(
    X, PL, max_iter = 2, tol = 1e-12,
    n_starts = 1, reg = 1e-6, covariance = "diagonal"
  )
  dense_size <- as.numeric(object.size(matrix(0, d, d)))

  expect_s4_class(fit$Sigma, "diagonalMatrix")
  expect_lt(as.numeric(object.size(fit$Sigma)), dense_size / 20)
  expect_equal(dim(fit$Sigma), c(d, d))
  expect_true(all(Matrix::diag(fit$Sigma) > 0))
  expect_identical(
    fit$diagnostics$covariance$storage, "sparse_diagonal"
  )
})

test_that("evidential estimators reject undefined inputs and controls", {
  dat <- soft_lda_fixture()
  bad_X <- dat$X
  bad_X[1, 1] <- Inf
  bad_PL <- dat$C
  bad_PL[1, 1] <- NA_real_
  zero_row <- dat$C
  zero_row[1, ] <- 0
  zero_column <- dat$C
  zero_column[, 3] <- 0

  expect_error(em_soft_lda(bad_X, dat$C), "finite")
  expect_error(em_soft_lda(dat$X, bad_PL), "finite")
  expect_error(em_soft_lda(dat$X, zero_row), "Every row")
  expect_error(em_soft_lda(dat$X, zero_column), "Every column")
  expect_error(em_soft_lda(dat$X, dat$C, max_iter = 0), "positive integer")
  expect_error(em_soft_lda(dat$X, dat$C, n_starts = 0), "positive integer")
  expect_error(em_soft_lda(dat$X, dat$C, tol = 0), "greater than")
  expect_error(em_soft_lda(dat$X, dat$C, reg = -1), "at least")
  expect_error(em_soft_lda(dat$X, dat$C, verbose = NA), "TRUE or FALSE")
  expect_error(
    em_soft_lda(dat$X, dat$C, covariance = "low_rank"),
    "arg.*one of"
  )
  expect_error(
    em_soft_lda(dat$X[1:2, , drop = FALSE], dat$C[1:2, , drop = FALSE]),
    "at least as many rows"
  )
  expect_error(soft_lr(dat$X, dat$C, lambda = -1), "at least")
  expect_error(soft_lr(dat$X, zero_column), "Every column")
})

test_that("binary soft-label logistic regression matches a direct ridge oracle", {
  x1 <- seq(-2.5, 2.5, length.out = 40)
  X <- cbind(x1 = x1, wave = sin(2 * x1))
  y <- ifelse(x1 + 0.65 * sin(2 * x1) > 0, 1L, 2L)
  PL <- em_test_crisp_labels(y, 2)
  lambda <- 0.4

  set.seed(1301)
  fit <- soft_lr(
    X, PL, max_iter = 200, tol = 1e-12,
    lambda = lambda, n_starts = 1
  )
  reference <- stats::optim(
    rep(0, ncol(X) + 1L),
    function(value) {
      beta <- matrix(value, ncol(X) + 1L, 1L)
      -em_test_lr_objective(beta, X, PL, lambda)
    },
    method = "BFGS",
    control = list(maxit = 2000, reltol = 1e-13)
  )
  fit_state <- em_test_lr_state(fit$beta, X, PL)

  expect_identical(reference$convergence, 0L)
  expect_gt(fit$iter, 1L)
  expect_equal(
    em_test_lr_objective(fit$beta, X, PL, lambda),
    -reference$value, tolerance = 2e-8
  )
  expect_equal(as.vector(fit$beta), reference$par, tolerance = 2e-5)
  expect_equal(unname(fit$zeta), fit_state$zeta, tolerance = 1e-11)
  expect_equal(fit$loglik, fit_state$loglik, tolerance = 1e-10)
})

test_that("multiclass logistic Hessian converges to the direct ridge oracle", {
  dat <- soft_lda_fixture()
  y <- rep(seq_len(3), each = 3)
  PL <- em_test_crisp_labels(y, 3)
  lambda <- 0.25
  parameter_count <- (ncol(dat$X) + 1L) * 2L

  set.seed(1302)
  fit <- soft_lr(
    dat$X, PL, max_iter = 300, tol = 1e-12,
    lambda = lambda, n_starts = 1
  )
  reference <- stats::optim(
    rep(0, parameter_count),
    function(value) {
      beta <- matrix(value, ncol(dat$X) + 1L, 2L)
      -em_test_lr_objective(beta, dat$X, PL, lambda)
    },
    method = "BFGS",
    control = list(maxit = 4000, reltol = 1e-13)
  )

  expect_identical(reference$convergence, 0L)
  expect_gt(fit$iter, 1L)
  expect_equal(
    em_test_lr_objective(fit$beta, dat$X, PL, lambda),
    -reference$value, tolerance = 5e-8
  )
  expect_equal(as.vector(fit$beta), reference$par, tolerance = 5e-5)
})

test_that("soft-label logistic regression returns a stationary consistent state", {
  dat <- soft_lda_fixture()
  lambda <- 0.1
  set.seed(1303)
  fit <- soft_lr(
    dat$X, dat$C, max_iter = 500, tol = 1e-12,
    lambda = lambda, n_starts = 1
  )
  state <- em_test_lr_state(fit$beta, dat$X, dat$C)
  Xtilde <- cbind(1, dat$X)
  gradient <- crossprod(
    Xtilde,
    state$zeta[, 1:2, drop = FALSE] -
      state$probability[, 1:2, drop = FALSE]
  ) - lambda * fit$beta

  expect_equal(unname(fit$zeta), unname(state$zeta), tolerance = 1e-10)
  expect_equal(fit$loglik, state$loglik, tolerance = 1e-9)
  expect_lt(max(abs(gradient)), 2e-5)
  expect_equal(rowSums(fit$zeta), rep(1, nrow(dat$X)), tolerance = 1e-12)
})

test_that("logistic E2M is monotone and invariant to plausibility scale", {
  dat <- soft_lda_fixture()
  lambda <- 0.1
  objectives <- vapply(seq_len(10), function(iterations) {
    set.seed(1304)
    fit <- soft_lr(
      dat$X, dat$C, max_iter = iterations, tol = 1e-14,
      lambda = lambda, n_starts = 1
    )
    em_test_lr_objective(fit$beta, dat$X, dat$C, lambda)
  }, numeric(1))
  row_scale <- exp(seq(-500, 500, length.out = nrow(dat$X)))
  scaled_PL <- sweep(dat$C, 1L, row_scale, FUN = "*")
  set.seed(1305)
  baseline <- soft_lr(
    dat$X, dat$C, max_iter = 300, tol = 1e-11,
    lambda = lambda, n_starts = 1
  )
  set.seed(1305)
  scaled <- soft_lr(
    dat$X, scaled_PL, max_iter = 300, tol = 1e-11,
    lambda = lambda, n_starts = 1
  )

  expect_true(all(diff(objectives) >= -1e-10))
  expect_gt(objectives[length(objectives)], objectives[1])
  expect_equal(scaled$beta, baseline$beta, tolerance = 1e-9)
  expect_equal(scaled$zeta, baseline$zeta, tolerance = 1e-9)
  expect_equal(
    scaled$loglik - baseline$loglik,
    sum(log(row_scale)), tolerance = 1e-8
  )
})

test_that("logistic E2M remains finite for extreme but finite logits", {
  X <- rbind(
    c(-1e6, 0), c(-8e5, 1), c(0, -1),
    c(1, 0), c(8e5, -1), c(1e6, 0)
  )
  PL <- rbind(
    c(1, .2, .2), c(1, .2, .2), c(.2, 1, .2),
    c(.2, 1, .2), c(.2, .2, 1), c(.2, .2, 1)
  )
  set.seed(1306)
  fit <- soft_lr(
    X, PL, max_iter = 20, tol = 1e-10,
    lambda = 1e-2, n_starts = 1
  )

  expect_true(all(is.finite(unlist(fit))))
  expect_equal(rowSums(fit$zeta), rep(1, nrow(X)), tolerance = 1e-12)
})
