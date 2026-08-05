library(testthat)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

make_gaussians <- function(n_per = 30, d = 6, n_classes = 3, sep = 3, sd = 1,
                           seed = 1) {
  set.seed(seed)
  X <- do.call(rbind, lapply(seq_len(n_classes), function(c) {
    mu <- rep(0, d)
    mu[((c - 1) %% d) + 1] <- sep * c
    matrix(rnorm(n_per * d, sd = sd), n_per, d) +
      matrix(mu, n_per, d, byrow = TRUE)
  }))
  y <- factor(rep(seq_len(n_classes), each = n_per))
  list(X = X, y = y)
}

expect_dgpage_graph_contract <- function(S, D, y, tol = 1e-12) {
  same <- outer(y, y, "==")
  off <- !diag(TRUE, length(y))

  expect_equal(dim(S), c(length(y), length(y)))
  expect_equal(dim(D), c(length(y), length(y)))
  expect_true(all(is.finite(S)))
  expect_true(all(is.finite(D)))
  expect_equal(S, t(S), tolerance = tol)
  expect_equal(D, t(D), tolerance = tol)
  expect_equal(diag(S), rep(0, length(y)), tolerance = tol)
  expect_equal(diag(D), rep(0, length(y)), tolerance = tol)
  expect_true(all(S[same & off] > -tol))
  expect_true(all(S[!same] < tol))
  expect_true(all(D[off] > 0))
}

reconstruct_dgpage_objective <- function(fit, S, D, alpha, beta) {
  Z <- fit$Z
  Y <- crossprod(fit$P, Z)
  W <- fit$W
  graph_quad <- function(Y, G) sum((Y %*% G) * Y)
  G_W <- .pairwise_laplacian(W)
  G_S <- .pairwise_laplacian(S)

  sum((Y - Y %*% W)^2) +
    alpha * sum((D %*% W)^2) +
    beta * graph_quad(Y, G_W) +
    (1 - alpha - beta) * graph_quad(Y, G_S)
}

dgpage_subspace_projector <- function(P) {
  Q <- qr.Q(qr(P))
  tcrossprod(Q)
}

closed_form_dgpage_W <- function(P, Z, D, alpha, beta) {
  Y <- crossprod(P, Z)
  YtY <- crossprod(Y)
  col_norms <- colSums(Y^2)
  E <- outer(col_norms, col_norms, "+") - 2 * YtY
  solve(YtY + alpha * crossprod(D), YtY - (beta / 2) * E)
}

# Literal double-loop implementations of Eqs. (4)-(7), kept as a slow
# reference to pin down the vectorized versions.
ref_similarity_graph <- function(X, y, q = 1.5) {
  n <- nrow(X)
  distMat <- as.matrix(dist(X))^2
  tau_plus <- tau_minus <- numeric(n)
  for (i in seq_len(n)) {
    same_idx <- which(y == y[i])
    diff_idx <- which(y != y[i])
    tau_plus[i] <- sum(distMat[i, same_idx]) / length(same_idx)^q
    tau_minus[i] <- sum(distMat[i, diff_idx]) / length(diff_idx)^q
  }
  S <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) next
      if (y[i] == y[j]) {
        S[i, j] <- 0.5 * (exp(-distMat[i, j] / tau_plus[i]) +
                            exp(-distMat[j, i] / tau_plus[j]))
      } else {
        S[i, j] <- -0.5 * (exp(-distMat[i, j] / tau_minus[i]) +
                             exp(-distMat[j, i] / tau_minus[j]))
      }
    }
  }
  S
}

ref_diversity_graph <- function(X, y) {
  n <- nrow(X)
  distMat <- as.matrix(dist(X))^2
  rho_plus <- rho_minus <- numeric(n)
  for (i in seq_len(n)) {
    rho_plus[i] <- sum(distMat[i, which(y == y[i])])
    rho_minus[i] <- sum(distMat[i, which(y != y[i])])
  }
  D <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) next
      rp_i <- if (y[i] == y[j]) rho_plus[i] else rho_minus[i]
      rp_j <- if (y[i] == y[j]) rho_plus[j] else rho_minus[j]
      D[i, j] <- 0.5 * (exp(distMat[i, j] / rp_i) + exp(distMat[j, i] / rp_j))
    }
  }
  D
}

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

test_that("compute_similarity_graph matches the literal Eq. (4) reference", {
  dat <- make_gaussians(n_per = 12, d = 4, n_classes = 3, seed = 11)
  S <- compute_similarity_graph(dat$X, dat$y, q = 1.5)
  S_ref <- ref_similarity_graph(dat$X, dat$y, q = 1.5)
  expect_equal(S, S_ref, tolerance = 1e-12)
})

test_that("compute_diversity_graph matches the literal Eq. (7) reference", {
  dat <- make_gaussians(n_per = 12, d = 4, n_classes = 3, seed = 12)
  D <- compute_diversity_graph(dat$X, dat$y)
  D_ref <- ref_diversity_graph(dat$X, dat$y)
  expect_equal(D, D_ref, tolerance = 1e-12)
})

test_that("graphs have the documented structure", {
  dat <- make_gaussians(n_per = 10, d = 4, n_classes = 2, seed = 13)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  expect_dgpage_graph_contract(S, D, dat$y)
})

test_that("graphs are invariant to translation, scale, and row permutation", {
  dat <- make_gaussians(n_per = 6, d = 4, n_classes = 3, seed = 14)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  shift <- matrix(seq_len(ncol(dat$X)), nrow(dat$X), ncol(dat$X), byrow = TRUE)
  X_shift_scale <- 3.7 * dat$X + shift
  expect_equal(compute_similarity_graph(X_shift_scale, dat$y), S, tolerance = 1e-12)
  expect_equal(compute_diversity_graph(X_shift_scale, dat$y), D, tolerance = 1e-12)

  permutation <- sample(seq_len(nrow(dat$X)))
  S_perm <- compute_similarity_graph(dat$X[permutation, , drop = FALSE], dat$y[permutation])
  D_perm <- compute_diversity_graph(dat$X[permutation, , drop = FALSE], dat$y[permutation])
  expect_equal(S_perm, S[permutation, permutation], tolerance = 1e-12)
  expect_equal(D_perm, D[permutation, permutation], tolerance = 1e-12)
})

test_that("graphs remain invariant at extreme finite measurement scales", {
  dat <- make_gaussians(n_per = 6, d = 4, n_classes = 3, seed = 140)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  for (scale in c(1e-150, 1e-8, 1e8, 1e150)) {
    expect_equal(compute_similarity_graph(scale * dat$X, dat$y), S, tolerance = 1e-12)
    expect_equal(compute_diversity_graph(scale * dat$X, dat$y), D, tolerance = 1e-12)
  }
})

test_that("graphs remain finite for duplicate and zero-distance samples", {
  X <- matrix(
    c(
      0, 0,
      0, 0,
      1, 1,
      1, 1
    ),
    ncol = 2,
    byrow = TRUE
  )
  y <- factor(c("a", "a", "b", "b"))

  S <- compute_similarity_graph(X, y)
  D <- compute_diversity_graph(X, y)

  expect_dgpage_graph_contract(S, D, y)
  expect_equal(S[1, 2], 1, tolerance = 1e-12)
  expect_equal(S[1, 3], -exp(-sqrt(2)), tolerance = 1e-12)
  expect_equal(D[1, 2], 1, tolerance = 1e-12)
  expect_equal(D[1, 3], exp(1 / 2), tolerance = 1e-12)
})

test_that("graph construction validates labels", {
  X <- matrix(rnorm(20 * 3), 20, 3)
  expect_error(compute_similarity_graph(X, rep("a", 20)), "two classes")
  expect_error(compute_similarity_graph(X, c(rep("a", 19), "b")), "at least two samples")
  expect_error(compute_similarity_graph(X, rep("a", 10)), "match nrow")
  expect_error(compute_similarity_graph(X, rep(c("a", "b"), each = 10), q = 0), "q must be")
  expect_error(compute_similarity_graph(cbind(X, Inf), rep(c("a", "b"), each = 10)), "finite")
  expect_error(compute_diversity_graph(X, rep("a", 20)), "two classes")
})

test_that("pairwise Laplacian matches the weighted pairwise-distance oracle", {
  set.seed(15)
  W <- matrix(runif(25), nrow = 5)
  diag(W) <- 0
  Y <- matrix(rnorm(15), nrow = 3)
  G <- .pairwise_laplacian(W)

  pairwise_sum <- 0
  for (i in seq_len(ncol(Y))) {
    for (j in seq_len(ncol(Y))) {
      pairwise_sum <- pairwise_sum + W[i, j] * sum((Y[, i] - Y[, j])^2)
    }
  }

  expect_equal(sum((Y %*% G) * Y), pairwise_sum, tolerance = 1e-10)
  expect_equal(rowSums(G), rep(0, nrow(G)), tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# Optimizer behavior
# ---------------------------------------------------------------------------

test_that("dgpage_fit actually iterates and the objective is non-increasing", {
  dat <- make_gaussians(n_per = 15, d = 5, n_classes = 3, seed = 21)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  fit <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 15, tol = 1e-12, seed = 5)

  # The convergence bug used to stop everything after 1 iteration
  expect_gt(fit$iter, 1)
  expect_equal(length(fit$obj_trace), fit$iter)
  expect_equal(fit$objective, fit$obj_trace[fit$iter])

  # Alternating minimization of a single objective must be monotone
  # (after the first P-update); allow tiny relative numerical slack.
  tr <- fit$obj_trace
  slack <- 1e-8 * pmax(1, abs(tr[-length(tr)]))
  expect_true(all(diff(tr) <= slack))
})

test_that("dgpage_fit respects maxiter and tol", {
  dat <- make_gaussians(n_per = 10, d = 4, n_classes = 2, seed = 22)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  fit3 <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 3, tol = 1e-15, seed = 5)
  expect_equal(fit3$iter, 3)

  # A loose tolerance must stop earlier than a strict one
  fit_loose <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 50, tol = 1e-2, seed = 5)
  fit_strict <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 50, tol = 1e-10, seed = 5)
  expect_lte(fit_loose$iter, fit_strict$iter)

  expect_error(dgpage_fit(dat$X, S, D, r = 2, maxiter = 0), "maxiter")
  expect_error(dgpage_fit(dat$X, S, D, r = 2, alpha = 0.6, beta = 0.5), "alpha")
  expect_error(dgpage_fit(dat$X, S, D, r = 0), "r must be a finite positive integer")
  expect_error(dgpage_fit(dat$X, S, D, r = Inf), "r must be a finite positive integer")
  expect_error(dgpage_fit(dat$X, S, D, r = 2, alpha = 0, beta = 1e-3), "alpha must be a finite positive")
  expect_error(dgpage_fit(dat$X, S, D, r = 2, maxiter = Inf), "maxiter must be a finite positive integer")
  expect_error(dgpage_fit(dat$X, S, D, r = 2, tol = Inf), "tol must be a finite positive")
  expect_error(dgpage_fit(dat$X, S, D, r = 2, ridge = Inf), "ridge must be a finite non-negative")
  expect_error(dgpage_fit(dat$X, S, D, r = 2, seed = Inf), "seed must be NULL or a finite integer")
  expect_error(dgpage_fit(dat$X, S, D[-1, ], r = 2), "D must be an n x n matrix")
  expect_error(dgpage_fit(cbind(dat$X, NA_real_), S, D, r = 2), "finite")
})

test_that("the W subproblem must be strictly convex", {
  dat <- make_gaussians(n_per = 8, d = 4, n_classes = 2, seed = 220)
  S <- compute_similarity_graph(dat$X, dat$y)
  singular_D <- matrix(0, nrow(dat$X), nrow(dat$X))

  expect_error(
    dgpage_fit(dat$X, S, singular_D, r = 2, alpha = 1e-3, beta = 1e-4,
               maxiter = 2, seed = 5),
    "W subproblem is not positive definite"
  )
})

test_that("P satisfies the generalized-eigenproblem constraint t(P) N P = I", {
  dat <- make_gaussians(n_per = 15, d = 5, n_classes = 2, seed = 23)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  fit <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 5, seed = 5)
  N <- tcrossprod(fit$Z)
  gram <- t(fit$P) %*% N %*% fit$P
  # ridge is 1e-8 relative, so the constraint holds to ~1e-6
  expect_equal(gram, diag(2), tolerance = 1e-5)
})

test_that("reported objective equals the DGPAGE objective formula", {
  alpha <- 1e-3
  beta <- 1e-4
  dat <- make_gaussians(n_per = 8, d = 4, n_classes = 2, seed = 26)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  fit <- dgpage_fit(dat$X, S, D, r = 2, alpha = alpha, beta = beta,
                    maxiter = 4, seed = 5)

  expect_true(all(is.finite(fit$W)))
  expect_equal(dim(fit$W), c(nrow(dat$X), nrow(dat$X)))
  expect_equal(reconstruct_dgpage_objective(fit, S, D, alpha, beta),
               fit$objective,
               tolerance = 1e-8)
})

test_that("returned W is optimal for the returned P", {
  alpha <- 1e-3
  beta <- 1e-4
  dat <- make_gaussians(n_per = 10, d = 5, n_classes = 3, seed = 260)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  fit <- dgpage_fit(dat$X, S, D, r = 2, alpha = alpha, beta = beta,
                    maxiter = 1, tol = 1e-15, seed = 5)
  expected_W <- closed_form_dgpage_W(fit$P, fit$Z, D, alpha, beta)

  expect_equal(fit$W, expected_W, tolerance = 1e-8)
  expect_false(fit$converged)
  expect_true(is.finite(fit$final_rel_change))
})

test_that("dgpage_fit is invariant to global measurement units", {
  dat <- make_gaussians(n_per = 10, d = 5, n_classes = 3, seed = 270)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  fit <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 12, tol = 1e-12, seed = 5)
  for (scale in c(1e-8, 1e-3, 1e3, 1e8)) {
    scaled <- dgpage_fit(scale * dat$X, S, D, r = 2,
                         maxiter = 12, tol = 1e-12, seed = 5)
    expect_equal(scaled$W, fit$W, tolerance = 1e-8)
    expect_equal(scaled$obj_trace, fit$obj_trace, tolerance = 1e-8)
    expect_equal(
      dgpage_subspace_projector(scaled$P),
      dgpage_subspace_projector(fit$P),
      tolerance = 1e-8
    )
    expect_equal(abs((scale * dat$X) %*% scaled$P),
                 abs(dat$X %*% fit$P), tolerance = 1e-8)
  }
})

test_that("seed argument gives reproducible fits without clobbering the RNG", {
  dat <- make_gaussians(n_per = 10, d = 4, n_classes = 2, seed = 24)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)

  f1 <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 3, seed = 7)
  f2 <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 3, seed = 7)
  expect_equal(f1$P, f2$P)

  # The caller's RNG state must be untouched when seed is supplied
  set.seed(42)
  state_before <- .Random.seed
  invisible(dgpage_fit(dat$X, S, D, r = 2, maxiter = 2, seed = 7))
  expect_identical(.Random.seed, state_before)

  # Without a seed, the global stream is used (not reset to a hardcoded seed):
  # downstream draws must depend on the caller's seed.
  set.seed(1)
  invisible(dgpage_fit(dat$X, S, D, r = 2, maxiter = 2))
  a <- rnorm(1)
  set.seed(2)
  invisible(dgpage_fit(dat$X, S, D, r = 2, maxiter = 2))
  b <- rnorm(1)
  expect_false(isTRUE(all.equal(a, b)))
})

test_that("randomized small fits preserve finite outputs and objective traces", {
  for (seed in 101:103) {
    dat <- make_gaussians(n_per = 5, d = 4, n_classes = 3, sep = 2, seed = seed)
    S <- compute_similarity_graph(dat$X, dat$y)
    D <- compute_diversity_graph(dat$X, dat$y)
    fit <- dgpage_fit(dat$X, S, D, r = 2, maxiter = 3, seed = seed)

    expect_true(all(is.finite(fit$P)))
    expect_true(all(is.finite(fit$W)))
    expect_true(all(is.finite(fit$obj_trace)))
    expect_equal(length(fit$obj_trace), fit$iter)
    expect_equal(fit$objective, fit$obj_trace[fit$iter])
  }
})

test_that("dgpage_fit solves in the numerical row space when d > n", {
  set.seed(31)
  n <- 14
  d <- 40
  X <- matrix(rnorm(n * d), n, d) +
    matrix(rep(c(0, 4), each = n / 2), n, d) # separated along all dims
  y <- factor(rep(1:2, each = n / 2))
  S <- compute_similarity_graph(X, y)
  D <- compute_diversity_graph(X, y)

  w <- capture_warnings(fit <- dgpage_fit(X, S, D, r = 20, maxiter = 3, seed = 5))
  expect_true(any(grepl("numerical rank", w)))
  expect_true(all(is.finite(fit$P)))
  expect_equal(ncol(fit$P), fit$effective_r)
  expect_equal(fit$effective_r, fit$optimization_rank)
  expect_lte(fit$optimization_rank, n)
  expect_equal(qr(X %*% fit$P)$rank, fit$effective_r)
  expect_equal(fit$requested_r, 20)
})

test_that("r above numerical rank is truncated with a warning", {
  dat <- make_gaussians(n_per = 10, d = 3, n_classes = 2, seed = 25)
  S <- compute_similarity_graph(dat$X, dat$y)
  D <- compute_diversity_graph(dat$X, dat$y)
  expect_warning(fit <- dgpage_fit(dat$X, S, D, r = 5, maxiter = 2, seed = 5),
                 "cannot exceed the numerical rank")
  expect_equal(ncol(fit$P), 3)
})

# ---------------------------------------------------------------------------
# Prediction on held-out data
# ---------------------------------------------------------------------------

test_that("dgpage_discriminant generalizes to held-out data", {
  dat <- make_gaussians(n_per = 40, d = 6, n_classes = 3, sep = 3, seed = 41)
  n <- nrow(dat$X)
  set.seed(42)
  test_idx <- unlist(lapply(split(seq_len(n), dat$y), sample, size = 10))
  train_idx <- setdiff(seq_len(n), test_idx)

  dp <- dgpage_discriminant(dat$X[train_idx, ], dat$y[train_idx],
                            r = 2, maxiter = 10, seed = 5)

  # Held-out accuracy: nearest neighbor is NOT the query itself here,
  # so this genuinely tests the learned projection.
  preds_1nn <- predict(dp, new_data = dat$X[test_idx, ], method = "1nn")
  expect_s3_class(preds_1nn, "factor")
  expect_gt(mean(preds_1nn == dat$y[test_idx]), 0.85)

  preds_lda <- predict(dp, new_data = dat$X[test_idx, ], method = "lda", type = "class")
  expect_gt(mean(preds_lda == dat$y[test_idx]), 0.85)

  # k > 1 does an actual majority vote and keeps factor levels
  preds_k3 <- predict(dp, new_data = dat$X[test_idx, ], method = "1nn", k = 3)
  expect_s3_class(preds_k3, "factor")
  expect_identical(levels(preds_k3), levels(dat$y))
  expect_gt(mean(preds_k3 == dat$y[test_idx]), 0.85)

  # prob output for lda
  probs <- predict(dp, new_data = dat$X[test_idx, ], method = "lda", type = "prob")
  expect_equal(dim(probs), c(length(test_idx), nlevels(dat$y)))
})

test_that("dgpage_predict works with only ytrain stored on the model", {
  dat <- make_gaussians(n_per = 30, d = 5, n_classes = 2, sep = 4, seed = 43)
  n <- nrow(dat$X)
  set.seed(44)
  test_idx <- sample(n, 15)
  train_idx <- setdiff(seq_len(n), test_idx)

  Xtr <- dat$X[train_idx, ]
  ytr <- dat$y[train_idx]
  S <- compute_similarity_graph(Xtr, ytr)
  D <- compute_diversity_graph(Xtr, ytr)
  model <- dgpage_fit(Xtr, S, D, r = 2, maxiter = 8, seed = 5)

  # The documented workflow: store labels, then predict
  expect_error(dgpage_predict(model, dat$X[test_idx, ]), "ytrain")
  model$ytrain <- ytr

  preds <- dgpage_predict(model, dat$X[test_idx, ])
  expect_s3_class(preds, "factor")
  expect_length(preds, length(test_idx))
  expect_gt(mean(preds == dat$y[test_idx]), 0.85)

  # k > 1 majority vote
  preds3 <- dgpage_predict(model, dat$X[test_idx, ], k = 3)
  expect_gt(mean(preds3 == dat$y[test_idx]), 0.85)

  expect_error(dgpage_predict(model, dat$X[test_idx, -1, drop = FALSE]), "same number of columns")
  X_nonfinite <- dat$X[test_idx, ]
  X_nonfinite[1, 1] <- Inf
  expect_error(dgpage_predict(model, X_nonfinite), "finite")
  expect_error(dgpage_predict(model, dat$X[test_idx, ], k = nrow(Xtr) + 1), "k must be")
  model_bad_labels <- model
  model_bad_labels$ytrain <- ytr[-1]
  expect_error(dgpage_predict(model_bad_labels, dat$X[test_idx, ]), "Length of ytrain")
})

test_that("k-NN voting breaks ties by factor level order", {
  labs <- factor(c("b", "a", "b", "a"), levels = c("a", "b"))
  idx <- matrix(c(1, 2, 3, 4), nrow = 1)

  expect_identical(.knn_vote(labs, idx), factor("a", levels = c("a", "b")))
})

# ---------------------------------------------------------------------------
# Projector object structure
# ---------------------------------------------------------------------------

test_that("dgpage_discriminant returns a coherent projector object", {
  dat <- make_gaussians(n_per = 12, d = 5, n_classes = 2, seed = 51)
  n <- nrow(dat$X)

  dp <- dgpage_discriminant(dat$X, dat$y, r = 2, maxiter = 5, seed = 5)

  expect_s3_class(dp, "dgpage_projector")
  expect_s3_class(dp, "discriminant_projector")
  expect_equal(dim(dp$v), c(ncol(dat$X), 2))
  expect_equal(dim(dp$s), c(n, 2))
  expect_length(dp$sdev, 2)
  expect_equal(dim(dp$W), c(n, n))
  expect_true(is.numeric(dp$objective))
  expect_gte(dp$iter, 1)

  # Scores must equal the projection of the training data
  expect_equal(dp$s, dat$X %*% dp$v, tolerance = 1e-10)

  # print method runs
  expect_output(print(dp), "dgpage_projector")

  expect_true(is.logical(dp$converged))
  expect_true(is.finite(dp$final_rel_change))
  expect_true(is.finite(dp$ridge_effective))
  expect_equal(dp$requested_r, 2)
  expect_equal(dp$effective_r, 2)
})

test_that("dgpage_discriminant exposes ridge and validates 1nn prediction type", {
  dat <- make_gaussians(n_per = 10, d = 5, n_classes = 2, seed = 52)
  dp <- dgpage_discriminant(dat$X, dat$y, r = 2, ridge = 1e-6,
                            maxiter = 3, seed = 5)

  expect_equal(dp$ridge, 1e-6)
  expect_error(
    predict(dp, new_data = dat$X[1:3, ], method = "1nn", type = "prob"),
    "type = .class."
  )

  X_nonfinite <- dat$X[1:3, ]
  X_nonfinite[1, 1] <- NA_real_
  expect_error(predict(dp, new_data = X_nonfinite, method = "1nn"), "finite")
})
