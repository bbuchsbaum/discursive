scatter_test_fixture <- function() {
  X <- matrix(
    c(
      1, 4, 2,
      2, 0, 5,
      4, 3, 1,
      7, 2, 6,
      8, 5, 4,
      9, 1, 7
    ),
    ncol = 3,
    byrow = TRUE
  )
  y <- c("beta", "alpha", "beta", "gamma", "alpha", "gamma")
  list(X = X, y = y)
}

scatter_reference <- function(X, y) {
  classes <- unique(as.character(y))
  overall <- colMeans(X)
  total <- Reduce(`+`, lapply(seq_len(nrow(X)), function(i) {
    tcrossprod(X[i, ] - overall)
  }))
  within <- matrix(0, ncol(X), ncol(X))
  between <- matrix(0, ncol(X), ncol(X))

  for (class in classes) {
    rows <- which(as.character(y) == class)
    class_mean <- colMeans(X[rows, , drop = FALSE])
    for (i in rows) {
      within <- within + tcrossprod(X[i, ] - class_mean)
    }
    between <- between + length(rows) * tcrossprod(class_mean - overall)
  }

  list(total = total, within = within, between = between)
}

test_that("scatter kernels match direct algebra and decompose total scatter", {
  dat <- scatter_test_fixture()
  ref <- scatter_reference(dat$X, dat$y)
  y <- factor(dat$y, levels = c("unused", "gamma", "alpha", "beta"))

  expect_equal(total_scatter(dat$X, colMeans(dat$X)), ref$total, tolerance = 1e-12)
  expect_equal(pooled_scatter(dat$X, y), ref$within, tolerance = 1e-12)
  expect_equal(within_class_scatter(dat$X, y), ref$within, tolerance = 1e-12)
  expect_equal(between_class_scatter(dat$X, y), ref$between, tolerance = 1e-12)
  expect_equal(ref$total, ref$within + ref$between, tolerance = 1e-12)
})

test_that("scatter kernels obey permutation, translation, and scale laws", {
  dat <- scatter_test_fixture()
  shift <- c(1e4, -3e4, 2e4)
  scale <- 7.5
  permutation <- c(6, 2, 5, 1, 4, 3)

  base_within <- within_class_scatter(dat$X, dat$y)
  base_between <- between_class_scatter(dat$X, dat$y)
  transformed <- sweep(scale * dat$X, 2, shift, "+")

  expect_equal(
    within_class_scatter(transformed, dat$y),
    scale^2 * base_within,
    tolerance = 1e-10
  )
  expect_equal(
    between_class_scatter(transformed, dat$y),
    scale^2 * base_between,
    tolerance = 1e-10
  )
  expect_equal(
    within_class_scatter(dat$X[permutation, ], dat$y[permutation]),
    base_within,
    tolerance = 1e-12
  )
  expect_equal(
    between_class_scatter(dat$X[permutation, ], dat$y[permutation]),
    base_between,
    tolerance = 1e-12
  )
})

test_that("randomized scatter decompositions agree across shapes and class balances", {
  for (seed in seq_len(12)) {
    set.seed(seed)
    n <- sample(8:24, 1)
    p <- sample(1:9, 1)
    groups <- sample(2:min(5, n), 1)
    y <- rep(seq_len(groups), length.out = n)
    y <- sample(y)
    X <- matrix(rnorm(n * p, sd = 10^runif(1, -3, 3)), n, p)
    reference <- scatter_reference(X, y)

    observed_total <- total_scatter(X, colMeans(X))
    observed_within <- within_class_scatter(X, y)
    observed_between <- between_class_scatter(X, y)

    expect_equal(observed_total, reference$total, tolerance = 1e-10)
    expect_equal(observed_within, reference$within, tolerance = 1e-10)
    expect_equal(observed_between, reference$between, tolerance = 1e-10)
    expect_equal(observed_total, observed_within + observed_between, tolerance = 1e-10)
  }
})

test_that("degenerate scatter cases are finite and retain feature names", {
  X <- matrix(c(1, 2, 1, 2, 1, 2), ncol = 2, byrow = TRUE)
  colnames(X) <- c("left", "right")
  y <- rep("only", nrow(X))

  expect_equal(within_class_scatter(X, y), matrix(0, 2, 2,
                                                   dimnames = list(colnames(X), colnames(X))))
  expect_equal(between_class_scatter(X, y), matrix(0, 2, 2,
                                                    dimnames = list(colnames(X), colnames(X))))
  expect_true(all(is.finite(total_scatter(X, colMeans(X)))))
})

test_that("moving scatter averages aligned local feature windows", {
  X <- matrix(seq_len(30), nrow = 6, ncol = 5)
  centered <- sweep(X, 2, colMeans(X), "-")
  expected <- Reduce(`+`, lapply(seq_len(3), function(start) {
    crossprod(centered[, start:(start + 2), drop = FALSE])
  })) / 3

  expect_equal(moving_scatter(X, 3), unname(expected), tolerance = 1e-12)
  expect_equal(dim(moving_scatter(X, 3)), c(3L, 3L))
  expect_equal(
    moving_scatter(X, 5),
    unname(crossprod(centered)),
    tolerance = 1e-12
  )

  expect_error(moving_scatter(X, 3.5), "integer")
  expect_error(moving_scatter(X, 2), "odd integer")
  expect_error(moving_scatter(X, 7), "number of columns")
})

test_that("binary between scatter matches its rank-one closed form", {
  dat <- scatter_test_fixture()
  id1 <- c(1L, 3L)
  id2 <- c(4L, 6L)
  mean_difference <- colMeans(dat$X[id1, , drop = FALSE]) -
    colMeans(dat$X[id2, , drop = FALSE])
  expected <- length(id1) * length(id2) / (length(id1) + length(id2)) *
    tcrossprod(mean_difference)

  expect_equal(binary_between_scatter(dat$X, id1, id2), expected, tolerance = 1e-12)

  logical_id1 <- seq_len(nrow(dat$X)) %in% id1
  logical_id2 <- seq_len(nrow(dat$X)) %in% id2
  expect_equal(binary_between_scatter(dat$X, logical_id1, logical_id2), expected)
  expect_error(binary_between_scatter(dat$X, c(1, 3), c(3, 4)), "disjoint")
  expect_error(binary_between_scatter(dat$X, integer(), c(3, 4)), "non-empty")
})

test_that("pairwise scatter reuses the defining two-class statistics", {
  dat <- scatter_test_fixture()
  observed <- pairwise_within_scatter(dat$X, dat$y)
  classes <- unique(dat$y)
  pairs <- combn(classes, 2)

  expect_length(observed, ncol(pairs))
  for (i in seq_len(ncol(pairs))) {
    rows <- dat$y %in% pairs[, i]
    ref <- scatter_reference(dat$X[rows, , drop = FALSE], dat$y[rows])
    expect_equal(observed[[i]]$w, ref$within, tolerance = 1e-12)
    expect_equal(observed[[i]]$b, ref$between, tolerance = 1e-12)
  }
})

test_that("scatter helpers reject malformed or non-finite inputs", {
  dat <- scatter_test_fixture()

  expect_error(total_scatter(as.data.frame(dat$X), colMeans(dat$X)), "numeric matrix")
  expect_error(total_scatter(dat$X, c(1, 2)), "length")
  expect_error(total_scatter(dat$X, c(1, 2, Inf)), "finite")
  expect_error(within_class_scatter(dat$X, dat$y[-1]), "length")
  expect_error(between_class_scatter(dat$X, replace(dat$y, 1, NA)), "missing")
  expect_error(pairwise_within_scatter(dat$X, rep("one", nrow(dat$X))), "two classes")

  contaminated <- dat$X
  contaminated[2, 2] <- Inf
  expect_error(within_class_scatter(contaminated, dat$y), "finite")
})
