#' Partially Supervised LDA with Soft Labels (Evidential EM)
#'
#' Fits a Linear Discriminant Analysis (LDA) model using *soft* (plausibility)
#' labels and the Evidential EM (E²M) algorithm with optional regularisation
#' and multiple random restarts.
#'
#' @param X A numeric matrix `n x d`.
#' @param y A numeric matrix `n x K` of plausibility values for each class
#'   and instance. This is the soft-label matrix; in earlier versions it was
#'   called `PL`.
#' @param ncomp Integer. Number of discriminant directions to retain in the
#'   returned projector. If `NULL` (default), `K - 1` directions are kept.
#' @param max_iter Maximum number of E²M iterations. Default `100`.
#' @param tol Convergence tolerance on the log-likelihood. Default `1e-6`.
#' @param n_starts Number of random initialisations. Default `5`.
#' @param reg Ridge penalty added to the covariance matrix for numerical
#'   stability. Default `1e-9`.
#' @param verbose If `TRUE`, prints progress.
#'
#' @return A `discriminant_projector` (subclass `"em_soft_lda"`) with extra
#'   fields `pi` (class priors), `mu` (`K x d` class means), `Sigma` (pooled
#'   covariance), `zeta` (posterior class probabilities), `loglik`, and
#'   `iter`.
#'
#' @references
#' Quost, B., Denoeux, T., Li, S. (2017). "Parametric classification with
#' soft labels using the Evidential EM algorithm." *Advances in Data Analysis
#' and Classification*, 11(4), 659–690.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' n <- 100; d <- 2; K <- 3
#' X <- rbind(
#'   MASS::mvrnorm(n, c(1,0), diag(2)),
#'   MASS::mvrnorm(n, c(-1,1), diag(2)),
#'   MASS::mvrnorm(n, c(0,-1), diag(2))
#' )
#' truth <- c(rep(1,n), rep(2,n), rep(3,n))
#' y_soft <- matrix(0, 3*n, K)
#' for (i in 1:(3*n)) {
#'   if (runif(1) < 0.2) {
#'     alt <- sample(setdiff(1:K, truth[i]), 1)
#'     y_soft[i, truth[i]] <- 0.5
#'     y_soft[i, alt]      <- 0.5
#'   } else {
#'     y_soft[i, truth[i]] <- 1
#'   }
#' }
#' fit <- em_soft_lda(X, y_soft)
#' print(fit)
em_soft_lda <- function(X, y,
                        ncomp = NULL,
                        max_iter = 100,
                        tol = 1e-6,
                        n_starts = 5,
                        reg = 1e-9,
                        verbose = FALSE) {

  cl <- match.call()
  PL <- y

  if (!is.matrix(X)) stop("X must be a matrix.")
  if (!is.matrix(PL)) stop("y (soft-label matrix) must be a matrix.")
  n <- nrow(X); d <- ncol(X)
  if (nrow(PL) != n) stop("y must have same number of rows as X.")
  K <- ncol(PL)
  if (K < 2) stop("At least two classes are required.")
  if (any(PL < 0)) stop("y cannot contain negative values.")
  if (any(rowSums(PL) == 0)) stop("Each instance must have at least one positive plausibility.")

  gauss_den <- function(x, mu, Sigma_inv, logdetS) {
    diff <- x - matrix(mu, nrow(x), d, byrow = TRUE)
    d2 <- rowSums((diff %*% Sigma_inv) * diff)
    exp(-0.5 * d2) / ((2 * pi)^(d / 2) * exp(logdetS / 2))
  }

  best_res <- NULL
  best_ll  <- -Inf

  for (start_id in seq_len(n_starts)) {
    cl_km <- kmeans(X, centers = K, nstart = 1)
    pi_vec <- as.numeric(table(cl_km$cluster) / n)
    mu_mat <- cl_km$centers
    Sigma  <- cov(X) + reg * diag(d)
    Sigma_inv <- solve(Sigma)
    logdetS   <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)

    old_ll <- -Inf
    it <- 0
    zeta <- NULL

    for (it in 1:max_iter) {
      phi_mat <- matrix(0, n, K)
      for (k in 1:K) {
        phi_mat[, k] <- gauss_den(X, mu_mat[k, ], Sigma_inv, logdetS)
      }
      num <- PL * (matrix(pi_vec, n, K, byrow = TRUE) * phi_mat)
      denom_vec <- rowSums(num)
      zeta <- num / denom_vec

      nk <- colSums(zeta)
      pi_vec <- nk / n
      mu_mat <- (t(zeta) %*% X) / nk

      S <- matrix(0, d, d)
      for (k in 1:K) {
        diff <- X - matrix(mu_mat[k, ], n, d, byrow = TRUE)
        S <- S + t(diff) * zeta[, k] %*% diff
      }
      Sigma <- S / n + reg * diag(d)

      attempt <- 0
      while (TRUE) {
        detS <- determinant(Sigma, logarithm = TRUE)$modulus
        if (is.infinite(detS)) {
          attempt <- attempt + 1
          if (attempt > 10) stop("Cannot stabilize Sigma.")
          Sigma <- Sigma + reg * diag(d) * 10
        } else break
      }
      Sigma_inv <- solve(Sigma)
      logdetS   <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)

      ll <- sum(log(denom_vec))
      if (verbose) cat("Start", start_id, "Iter", it, "LL:", ll, "\n")
      if (abs(ll - old_ll) < tol) {
        old_ll <- ll
        break
      }
      old_ll <- ll
    }

    if (old_ll > best_ll) {
      best_ll <- old_ll
      best_res <- list(pi = pi_vec, mu = mu_mat, Sigma = Sigma,
                       zeta = zeta, loglik = old_ll, iter = it,
                       Sigma_inv = Sigma_inv)
    }
  }

  # Fisher discriminant directions from fitted (mu, Sigma):
  # eigen-decomp of Sigma_inv * Sb where Sb is between-class scatter from soft posteriors.
  pi_vec    <- best_res$pi
  mu_mat    <- best_res$mu
  Sigma     <- best_res$Sigma
  Sigma_inv <- best_res$Sigma_inv

  global_mu <- colSums(pi_vec * mu_mat)
  Sb <- matrix(0, d, d)
  for (k in seq_len(K)) {
    delta <- matrix(mu_mat[k, ] - global_mu, nrow = d, ncol = 1)
    Sb <- Sb + pi_vec[k] * tcrossprod(delta)
  }

  M <- Sigma_inv %*% Sb
  eig <- eigen(M, symmetric = FALSE)
  # take real parts and order by descending real eigenvalue
  vals <- Re(eig$values)
  vecs <- Re(eig$vectors)
  ord  <- order(vals, decreasing = TRUE)
  max_dim <- min(K - 1, d)
  use_dim <- if (is.null(ncomp)) max_dim else min(ncomp, max_dim)
  V <- vecs[, ord[seq_len(use_dim)], drop = FALSE]
  # normalise columns
  V <- apply(V, 2, function(col) col / sqrt(sum(col^2)))
  if (!is.matrix(V)) V <- matrix(V, nrow = d)

  s_mat <- X %*% V

  multivarious::discriminant_projector(
    v       = V,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    labels  = best_res$zeta,
    pi      = pi_vec,
    mu      = mu_mat,
    Sigma   = Sigma,
    zeta    = best_res$zeta,
    loglik  = best_res$loglik,
    iter    = best_res$iter,
    method  = "em_soft_lda",
    call    = cl,
    classes = "em_soft_lda"
  )
}


#' Partially Supervised Logistic Regression with Soft Labels (Evidential EM)
#'
#' Fits a logistic regression model using E²M with soft labels.
#'
#' @param X A numeric matrix `n x d`.
#' @param y A numeric matrix `n x K` of plausibilities.
#' @param max_iter Maximum E²M iterations. Default `100`.
#' @param tol Convergence tolerance on the log-likelihood. Default `1e-6`.
#' @param lambda Ridge regularisation parameter for coefficients. Default `1e-5`.
#' @param n_starts Number of initialisations. Default `3`.
#' @param verbose If `TRUE`, prints progress.
#'
#' @return A list with `beta`, `zeta`, `loglik`, and `iter`.
#'
#' @references
#' Quost, B., Denoeux, T., Li, S. (2017). "Parametric classification with
#' soft labels using the Evidential EM algorithm." *Advances in Data Analysis
#' and Classification*, 11(4), 659–690.
soft_lr <- function(X, y, max_iter = 100, tol = 1e-6,
                    lambda = 1e-5, n_starts = 3, verbose = FALSE) {
  PL <- y
  if (!is.matrix(X)) stop("X must be a matrix.")
  if (!is.matrix(PL)) stop("y must be a matrix.")
  n <- nrow(X); d <- ncol(X)
  if (nrow(PL) != n) stop("y and X must have same number of rows.")
  K <- ncol(PL)
  if (K < 2) stop("Need at least two classes for logistic regression.")
  if (any(PL < 0)) stop("y cannot have negative values.")
  if (any(rowSums(PL) == 0)) stop("Each instance must have at least one positive plausibility.")

  Xtilde <- cbind(1, X)

  softmax_prob <- function(eta) {
    denom <- 1 + rowSums(exp(eta))
    cbind(exp(eta) / denom, 1 / denom)
  }

  best_ll  <- -Inf
  best_res <- NULL

  for (start_id in seq_len(n_starts)) {
    beta <- matrix(rnorm((d + 1) * (K - 1), sd = 0.1), d + 1, K - 1)

    old_ll <- -Inf
    for (it in seq_len(max_iter)) {
      eta <- Xtilde %*% beta
      Pmat <- softmax_prob(eta)

      num <- PL * Pmat
      denom_vec <- rowSums(num)
      zeta <- num / denom_vec

      ll <- sum(log(denom_vec))
      if (verbose) cat("Start", start_id, "Iter", it, "LL:", ll, "\n")
      if (abs(ll - old_ll) < tol) {
        old_ll <- ll
        break
      }

      Y_star   <- zeta[, 1:(K - 1), drop = FALSE]
      diff_mat <- Y_star - Pmat[, 1:(K - 1), drop = FALSE]
      grad <- t(Xtilde) %*% diff_mat - lambda * cbind(beta)

      d1 <- (d + 1) * (K - 1)
      H <- matrix(0, d1, d1)
      for (i in seq_len(n)) {
        p_i <- Pmat[i, 1:(K - 1)]
        R_i <- diag(p_i) - outer(p_i, p_i)
        Xi <- matrix(Xtilde[i, ], d + 1, 1)
        H <- H - (Xi %*% t(Xi)) %x% R_i
      }
      H <- H - diag(lambda, d1)

      vec_beta <- as.vector(beta)
      vec_grad <- as.vector(grad)

      delta <- tryCatch(solve(H, vec_grad), error = function(e) {
        temp_lambda <- lambda * 10
        for (rtry in 1:5) {
          H2 <- H - diag(temp_lambda, d1)
          test <- try(solve(H2, vec_grad), silent = TRUE)
          if (!inherits(test, "try-error")) return(test)
          temp_lambda <- temp_lambda * 10
        }
        stop("Cannot invert Hessian even after increasing ridge.")
      })

      step <- 1
      repeat {
        new_vec  <- vec_beta + step * delta
        new_beta <- matrix(new_vec, d + 1, K - 1)
        eta_new  <- Xtilde %*% new_beta
        Pmat_new <- softmax_prob(eta_new)
        ll_new   <- sum(log(rowSums(PL * Pmat_new)))
        if (ll_new > old_ll || step < 1e-12) {
          beta   <- new_beta
          old_ll <- ll_new
          break
        } else {
          step <- step / 2
        }
      }
    }
    if (old_ll > best_ll) {
      eta <- Xtilde %*% beta
      Pmat <- softmax_prob(eta)
      num <- PL * Pmat
      denom_vec <- rowSums(num)
      zeta <- num / denom_vec
      best_res <- list(beta = beta, zeta = zeta, loglik = old_ll, iter = it)
      best_ll  <- old_ll
    }
  }
  best_res
}
