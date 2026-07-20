.disc_validate_positive_integer <- function(x, name) {
  valid <- is.numeric(x) && length(x) == 1L && is.finite(x) &&
    x >= 1 && x == floor(x) && x <= .Machine$integer.max
  if (!valid) {
    stop(name, " must be a positive integer.", call. = FALSE)
  }
  as.integer(x)
}

.disc_validate_scalar <- function(x, name, lower, strict = FALSE) {
  valid <- is.numeric(x) && length(x) == 1L && is.finite(x)
  if (strict) {
    valid <- valid && x > lower
  } else {
    valid <- valid && x >= lower
  }
  if (!valid) {
    relation <- if (strict) "greater than" else "at least"
    stop(name, " must be a finite scalar ", relation, " ", lower, ".",
         call. = FALSE)
  }
  as.numeric(x)
}

.disc_validate_verbose <- function(verbose) {
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("verbose must be TRUE or FALSE.", call. = FALSE)
  }
  verbose
}

.disc_validate_evidential_inputs <- function(X, PL, require_kmeans = FALSE) {
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("X must be a numeric matrix.", call. = FALSE)
  }
  if (!is.matrix(PL) || !is.numeric(PL)) {
    stop("PL must be a numeric matrix.", call. = FALSE)
  }
  if (nrow(X) < 1L || ncol(X) < 1L) {
    stop("X must have at least one row and one column.", call. = FALSE)
  }
  if (nrow(PL) != nrow(X)) {
    stop("PL must have the same number of rows as X.", call. = FALSE)
  }
  if (ncol(PL) < 2L) {
    stop("At least two classes are required.", call. = FALSE)
  }
  if (require_kmeans && nrow(X) < ncol(PL)) {
    stop("X must have at least as many rows as PL has classes.",
         call. = FALSE)
  }
  if (any(!is.finite(X))) {
    stop("X must contain only finite values.", call. = FALSE)
  }
  if (any(!is.finite(PL))) {
    stop("PL must contain only finite values.", call. = FALSE)
  }
  if (any(PL < 0)) {
    stop("PL cannot contain negative values.", call. = FALSE)
  }
  if (any(rowSums(PL > 0) == 0L)) {
    stop("Every row of PL must contain positive plausibility mass.",
         call. = FALSE)
  }
  if (any(colSums(PL > 0) == 0L)) {
    stop("Every column of PL must contain positive plausibility mass.",
         call. = FALSE)
  }

  storage.mode(X) <- "double"
  storage.mode(PL) <- "double"
  list(X = X, PL = PL)
}

.disc_factor_spd <- function(x, base_jitter = 0, label = "matrix",
                             max_attempts = 12L) {
  x <- (x + t(x)) / 2
  if (any(!is.finite(x))) {
    stop(label, " contains non-finite values.", call. = FALSE)
  }

  d <- nrow(x)
  scale <- max(abs(base::diag(x)))
  if (!is.finite(scale) || scale == 0) {
    scale <- 1
  }
  jitter <- 0

  for (attempt in 0:max_attempts) {
    candidate <- if (jitter == 0) x else x + base::diag(jitter, d)
    factor <- tryCatch(chol(candidate), error = function(e) NULL)
    if (!is.null(factor)) {
      return(list(
        matrix = candidate,
        chol = factor,
        logdet = 2 * sum(log(base::diag(factor))),
        jitter = jitter
      ))
    }
    if (attempt == max_attempts) {
      break
    }
    jitter <- if (jitter == 0) {
      max(base_jitter, sqrt(.Machine$double.eps) * scale)
    } else {
      jitter * 10
    }
  }

  stop("Could not obtain a positive-definite ", label,
       " after adaptive regularization.", call. = FALSE)
}

.disc_factor_restricted_covariance <- function(value, dimension, policy,
                                                base_jitter = 0,
                                                label = "covariance") {
  value <- as.numeric(value)
  if (any(!is.finite(value))) {
    stop(label, " contains non-finite values.", call. = FALSE)
  }
  scale <- max(abs(value))
  if (!is.finite(scale) || scale == 0) {
    scale <- 1
  }
  numerical_floor <- sqrt(.Machine$double.eps) * scale
  jitter <- 0
  if (min(value) < numerical_floor) {
    jitter <- max(base_jitter, numerical_floor - min(value))
  }
  stabilized <- value + jitter
  if (any(!is.finite(stabilized)) || any(stabilized <= 0)) {
    stop("Could not obtain a positive-definite ", label,
         " after adaptive regularization.", call. = FALSE)
  }

  logdet <- if (policy == "diagonal") {
    sum(log(stabilized))
  } else {
    dimension * log(stabilized)
  }
  list(
    policy = policy,
    variance = stabilized,
    logdet = logdet,
    jitter = jitter
  )
}

.disc_factor_covariance <- function(value, dimension, policy, base_jitter = 0,
                                    label = "covariance") {
  if (policy == "full") {
    factor <- .disc_factor_spd(
      value, base_jitter = base_jitter, label = label
    )
    factor$policy <- "full"
    return(factor)
  }
  .disc_factor_restricted_covariance(
    value,
    dimension = dimension,
    policy = policy,
    base_jitter = base_jitter,
    label = label
  )
}

.disc_covariance_value <- function(factor) {
  if (factor$policy == "full") factor$matrix else factor$variance
}

.disc_covariance_output <- function(factor, dimension) {
  if (factor$policy == "full") {
    covariance <- factor$matrix
  } else {
    variance <- if (factor$policy == "diagonal") {
      factor$variance
    } else {
      rep(factor$variance, dimension)
    }
    covariance <- Matrix::Diagonal(n = dimension, x = variance)
  }
  covariance
}

.disc_initial_covariance_value <- function(total_scatter, total_ss, n, d,
                                            policy, reg) {
  switch(
    policy,
    full = total_scatter / (n - 1) + base::diag(reg, d),
    diagonal = total_ss / (n - 1) + reg,
    spherical = sum(total_ss) / ((n - 1) * d) + reg
  )
}

.disc_m_step_covariance_value <- function(total_scatter, total_ss, mu, nk,
                                           n, d, policy, reg) {
  if (policy == "full") {
    within_scatter <- total_scatter - crossprod(mu * sqrt(nk))
    within_scatter <- (within_scatter + t(within_scatter)) / 2
    return(within_scatter / n + base::diag(reg, d))
  }
  between_ss <- colSums(mu^2 * nk)
  if (policy == "diagonal") {
    return((total_ss - between_ss) / n + reg)
  }
  (sum(total_ss) - sum(between_ss)) / (n * d) + reg
}

.disc_interpolate_covariance <- function(current_factor, candidate_factor,
                                          step, dimension, policy) {
  value <- (1 - step) * .disc_covariance_value(current_factor) +
    step * .disc_covariance_value(candidate_factor)
  trial <- .disc_factor_covariance(
    value, dimension = dimension, policy = policy,
    label = "backtracked covariance"
  )
  extra_jitter <- trial$jitter
  trial$jitter <- (1 - step) * current_factor$jitter +
    step * candidate_factor$jitter + extra_jitter
  trial
}

.disc_solve_spd <- function(chol_factor, rhs) {
  backsolve(chol_factor, forwardsolve(t(chol_factor), rhs))
}

.disc_normalize_log_rows <- function(log_weights) {
  row_max <- apply(log_weights, 1L, max)
  if (any(!is.finite(row_max))) {
    stop("At least one observation has zero numerical mass in every class.",
         call. = FALSE)
  }
  scaled <- exp(sweep(log_weights, 1L, row_max, FUN = "-"))
  totals <- rowSums(scaled)
  if (any(!is.finite(totals)) || any(totals <= 0)) {
    stop("Could not normalize class probabilities.", call. = FALSE)
  }
  list(
    weights = sweep(scaled, 1L, totals, FUN = "/"),
    log_norm = row_max + log(totals)
  )
}

.disc_lda_log_density <- function(X, mu, covariance_factor) {
  if (covariance_factor$policy == "full") {
    whitened_X <- t(forwardsolve(t(covariance_factor$chol), t(X)))
    whitened_mu <- t(forwardsolve(t(covariance_factor$chol), t(mu)))
  } else if (covariance_factor$policy == "diagonal") {
    scale <- sqrt(covariance_factor$variance)
    whitened_X <- sweep(X, 2L, scale, FUN = "/")
    whitened_mu <- sweep(mu, 2L, scale, FUN = "/")
  } else {
    scale <- sqrt(covariance_factor$variance)
    whitened_X <- X / scale
    whitened_mu <- mu / scale
  }
  squared_distance <- outer(
    rowSums(whitened_X^2), rowSums(whitened_mu^2), FUN = "+"
  ) - 2 * tcrossprod(whitened_X, whitened_mu)
  squared_distance <- pmax(squared_distance, 0)
  constant <- -0.5 * (
    ncol(X) * log(2 * base::pi) + covariance_factor$logdet
  )
  constant - 0.5 * squared_distance
}

.disc_lda_e_step <- function(X, log_PL, class_priors, mu, covariance_factor) {
  log_density <- .disc_lda_log_density(X, mu, covariance_factor)
  log_joint <- log_PL + sweep(
    log_density, 2L, log(class_priors), FUN = "+"
  )
  normalized <- .disc_normalize_log_rows(log_joint)
  loglik <- sum(normalized$log_norm)
  if (!is.finite(loglik)) {
    stop("The evidential log-likelihood is not finite.", call. = FALSE)
  }
  list(zeta = normalized$weights, loglik = loglik)
}

.disc_abort_lda_start <- function(reason, message) {
  condition <- structure(
    list(message = message, call = NULL, reason = reason),
    class = c("disc_lda_start_error", "error", "condition")
  )
  stop(condition)
}

.disc_run_lda_start <- function(start_id, centered_X, log_PL, loglik_offset,
                                total_scatter, total_ss, initial_factor,
                                max_iter, tol, reg, covariance, verbose) {
  n <- nrow(centered_X)
  d <- ncol(centered_X)
  K <- ncol(log_PL)
  clusters <- tryCatch(
    stats::kmeans(centered_X, centers = K, nstart = 1),
    error = function(e) {
      .disc_abort_lda_start(
        "initialization_failed",
        paste0("k-means initialization failed: ", conditionMessage(e))
      )
    }
  )
  current <- list(
    pi = tabulate(clusters$cluster, nbins = K) / n,
    mu = clusters$centers,
    factor = initial_factor
  )
  current_state <- .disc_lda_e_step(
    centered_X, log_PL, current$pi, current$mu, current$factor
  )
  history <- current_state$loglik + loglik_offset
  jitter_history <- initial_factor$jitter
  iteration_count <- 0L
  final_change <- NA_real_
  converged <- FALSE
  termination_reason <- "max_iter"

  for (it in seq_len(max_iter)) {
    nk <- colSums(current_state$zeta)
    if (any(nk <= n * .Machine$double.eps)) {
      .disc_abort_lda_start(
        "class_mass_collapse",
        "At least one class lost all usable posterior mass during EM."
      )
    }

    candidate_pi <- nk / n
    candidate_mu <- crossprod(current_state$zeta, centered_X) / nk
    candidate_covariance <- .disc_m_step_covariance_value(
      total_scatter, total_ss, candidate_mu, nk,
      n = n, d = d, policy = covariance, reg = reg
    )
    candidate_factor <- .disc_factor_covariance(
      candidate_covariance,
      dimension = d,
      policy = covariance,
      base_jitter = reg,
      label = paste(covariance, "common covariance")
    )
    jitter_history <- c(jitter_history, candidate_factor$jitter)
    candidate <- list(
      pi = candidate_pi,
      mu = candidate_mu,
      factor = candidate_factor
    )
    candidate_state <- .disc_lda_e_step(
      centered_X, log_PL, candidate$pi, candidate$mu, candidate$factor
    )

    monotonic_tolerance <- 100 * .Machine$double.eps *
      (1 + abs(current_state$loglik))
    accepted <- candidate_state$loglik >=
      current_state$loglik - monotonic_tolerance

    if (!accepted) {
      for (backtrack in seq_len(30L)) {
        step <- 0.5^backtrack
        trial_factor <- .disc_interpolate_covariance(
          current$factor, candidate$factor, step,
          dimension = d, policy = covariance
        )
        jitter_history <- c(jitter_history, trial_factor$jitter)
        trial <- list(
          pi = (1 - step) * current$pi + step * candidate$pi,
          mu = (1 - step) * current$mu + step * candidate$mu,
          factor = trial_factor
        )
        trial_state <- .disc_lda_e_step(
          centered_X, log_PL, trial$pi, trial$mu, trial$factor
        )
        if (trial_state$loglik >=
            current_state$loglik - monotonic_tolerance) {
          candidate <- trial
          candidate_state <- trial_state
          accepted <- TRUE
          break
        }
      }
    }

    if (!accepted) {
      termination_reason <- "line_search_failed"
      if (verbose) {
        cat("Start", start_id, "stalled without a monotone EM update\n")
      }
      break
    }

    previous_loglik <- current_state$loglik
    current <- candidate
    current_state <- candidate_state
    iteration_count <- it
    final_change <- current_state$loglik - previous_loglik
    history <- c(history, current_state$loglik + loglik_offset)

    if (verbose) {
      cat("Start", start_id, "Iter", it, "LL:",
          current_state$loglik + loglik_offset, "\n")
    }
    if (abs(final_change) <= tol * (1 + abs(previous_loglik))) {
      converged <- TRUE
      termination_reason <- "relative_loglik_tolerance"
      break
    }
  }

  list(
    start = start_id,
    pi = current$pi,
    mu = current$mu,
    factor = current$factor,
    state = current_state,
    loglik = current_state$loglik + loglik_offset,
    iter = iteration_count,
    converged = converged,
    termination_reason = termination_reason,
    final_change = final_change,
    history = history,
    initial_jitter = initial_factor$jitter,
    final_jitter = current$factor$jitter,
    max_jitter = max(jitter_history)
  )
}

.disc_multinomial_state <- function(eta) {
  if (any(!is.finite(eta))) {
    stop("The logistic linear predictors are not finite.", call. = FALSE)
  }
  row_max <- pmax(0, apply(eta, 1L, max))
  shifted <- exp(sweep(eta, 1L, row_max, FUN = "-"))
  scaled_denom <- exp(-row_max) + rowSums(shifted)
  log_denom <- row_max + log(scaled_denom)
  log_prob <- cbind(
    sweep(eta, 1L, log_denom, FUN = "-"),
    -log_denom
  )
  list(prob = exp(log_prob), log_prob = log_prob, log_denom = log_denom,
       eta = eta)
}

.disc_lr_e_step <- function(Xtilde, beta, log_PL) {
  model <- .disc_multinomial_state(Xtilde %*% beta)
  normalized <- .disc_normalize_log_rows(log_PL + model$log_prob)
  loglik <- sum(normalized$log_norm)
  if (!is.finite(loglik)) {
    stop("The evidential log-likelihood is not finite.", call. = FALSE)
  }
  c(model, list(zeta = normalized$weights, loglik = loglik))
}

.disc_lr_q <- function(model, zeta, beta, lambda) {
  q <- ncol(beta)
  sum(zeta[, seq_len(q), drop = FALSE] * model$eta) -
    sum(model$log_denom) - 0.5 * lambda * sum(beta^2)
}

.disc_lr_newton_system <- function(Xtilde, model, zeta, beta, lambda) {
  p <- nrow(beta)
  q <- ncol(beta)
  parameter_count <- p * q
  gradient_matrix <- crossprod(
    Xtilde,
    zeta[, seq_len(q), drop = FALSE] -
      model$prob[, seq_len(q), drop = FALSE]
  ) - lambda * beta
  gradient <- as.vector(gradient_matrix)

  information <- matrix(0, parameter_count, parameter_count)
  for (a in seq_len(q)) {
    ia <- (a - 1L) * p + seq_len(p)
    for (b in a:q) {
      ib <- (b - 1L) * p + seq_len(p)
      weight <- model$prob[, a] *
        ((a == b) - model$prob[, b])
      block <- crossprod(Xtilde, Xtilde * weight)
      information[ia, ib] <- block
      information[ib, ia] <- t(block)
    }
  }
  diagonal_index <- cbind(seq_len(parameter_count), seq_len(parameter_count))
  information[diagonal_index] <- information[diagonal_index] + lambda
  factor <- .disc_factor_spd(
    information, base_jitter = lambda, label = "logistic information matrix"
  )

  list(
    gradient = gradient,
    direction = as.vector(.disc_solve_spd(factor$chol, gradient))
  )
}

#' Evidential-EM Linear Discriminant Model with Soft Labels
#'
#' Fits a Gaussian linear discriminant model when each observation has a
#' vector of class plausibilities. This estimator is distinct from
#' [soft_lda()], which estimates a discriminant projection matrix.
#'
#' @param X Numeric matrix with observations in rows and features in columns.
#' @param PL Nonnegative numeric matrix with one plausibility row per
#'   observation and one column per class. Every row and column must have
#'   positive mass.
#' @param max_iter Positive integer maximum number of EM iterations per start.
#' @param tol Positive relative convergence tolerance for the evidential
#'   log-likelihood.
#' @param n_starts Positive integer number of k-means initializations.
#' @param reg Nonnegative ridge added to the common covariance matrix. If the
#'   result is still singular, a scale-aware numerical jitter is added.
#' @param verbose Logical; print iteration diagnostics when `TRUE`.
#' @param covariance Common-covariance policy. `"full"` estimates an
#'   unrestricted covariance matrix, `"diagonal"` estimates one variance per
#'   feature, and `"spherical"` estimates a single shared variance.
#'
#' @return A list containing class priors `pi`, class means `mu`, common
#'   covariance `Sigma`, posterior memberships `zeta`, final evidential
#'   log-likelihood `loglik`, iteration count `iter`, and `diagnostics`.
#'   For `covariance = "full"`, `Sigma` is a dense matrix. For the diagonal and
#'   spherical policies it is a memory-efficient [Matrix::Diagonal()] object.
#'   `diagnostics` contains:
#'   * `converged`, `termination_reason`, and `final_loglik_change` for the
#'     selected start;
#'   * `loglik_history`, including the initial and every accepted state;
#'   * `chosen_start` and a `starts` data frame with every start's outcome;
#'   * `covariance`, recording the policy, requested ridge, initial and final
#'     adaptive jitter, maximum jitter encountered, and storage type; and
#'   * `class_mapping`, which maps output classes to columns of `PL`.
#'
#' @details
#' Each covariance policy uses its exact constrained Gaussian M-step. The
#'   `reg` value is part of that M-step; `initial_jitter`, `final_jitter`, and
#'   `max_jitter` record only additional adaptive stabilization. A failed
#'   initialization or numerical failure is isolated to its start. The fit
#'   stops with an aggregated error only if every requested start fails.
#'
#' @references
#' Quost, B., Denoeux, T., and Li, S. (2017). Parametric classification with
#' soft labels using the Evidential EM algorithm. *Advances in Data Analysis
#' and Classification*, 11, 659--690.
#'
#' @seealso [soft_lda()]
#' @export
#'
#' @examples
#' set.seed(42)
#' X <- rbind(matrix(rnorm(40, -1), 20, 2), matrix(rnorm(40, 1), 20, 2))
#' PL <- cbind(c(rep(1, 20), rep(0.1, 20)),
#'             c(rep(0.1, 20), rep(1, 20)))
#' fit <- em_soft_lda(X, PL, n_starts = 1, max_iter = 20)
#' rowSums(fit$zeta)
em_soft_lda <- function(X, PL, max_iter = 100, tol = 1e-6, n_starts = 5,
                        reg = 1e-9, verbose = FALSE,
                        covariance = c("full", "diagonal", "spherical")) {
  validated <- .disc_validate_evidential_inputs(
    X, PL, require_kmeans = TRUE
  )
  X <- validated$X
  PL <- validated$PL
  max_iter <- .disc_validate_positive_integer(max_iter, "max_iter")
  n_starts <- .disc_validate_positive_integer(n_starts, "n_starts")
  tol <- .disc_validate_scalar(tol, "tol", 0, strict = TRUE)
  reg <- .disc_validate_scalar(reg, "reg", 0)
  verbose <- .disc_validate_verbose(verbose)
  covariance <- match.arg(covariance)

  n <- nrow(X)
  d <- ncol(X)
  K <- ncol(PL)
  feature_center <- colMeans(X)
  centered_X <- sweep(X, 2L, feature_center, FUN = "-")
  total_ss <- colSums(centered_X^2)
  total_scatter <- if (covariance == "full") {
    crossprod(centered_X)
  } else {
    NULL
  }
  raw_log_PL <- log(PL)
  PL_log_offset <- apply(raw_log_PL, 1L, max)
  log_PL <- sweep(raw_log_PL, 1L, PL_log_offset, FUN = "-")
  loglik_offset <- sum(PL_log_offset)

  initial_covariance <- .disc_initial_covariance_value(
    total_scatter, total_ss, n = n, d = d, policy = covariance, reg = reg
  )
  initial_factor <- .disc_factor_covariance(
    initial_covariance,
    dimension = d,
    policy = covariance,
    base_jitter = reg,
    label = paste("initial", covariance, "covariance")
  )

  start_outcomes <- vector("list", n_starts)
  best_result <- NULL
  best_loglik <- -Inf

  for (start_id in seq_len(n_starts)) {
    result <- tryCatch(
      .disc_run_lda_start(
        start_id = start_id,
        centered_X = centered_X,
        log_PL = log_PL,
        loglik_offset = loglik_offset,
        total_scatter = total_scatter,
        total_ss = total_ss,
        initial_factor = initial_factor,
        max_iter = max_iter,
        tol = tol,
        reg = reg,
        covariance = covariance,
        verbose = verbose
      ),
      error = function(e) e
    )
    if (inherits(result, "error")) {
      failure_reason <- if (inherits(result, "disc_lda_start_error")) {
        result$reason
      } else {
        "numerical_failure"
      }
      start_outcomes[[start_id]] <- data.frame(
        start = start_id,
        status = "failed",
        converged = FALSE,
        termination_reason = failure_reason,
        iter = NA_integer_,
        loglik = NA_real_,
        final_loglik_change = NA_real_,
        initial_jitter = initial_factor$jitter,
        final_jitter = NA_real_,
        max_jitter = NA_real_,
        message = conditionMessage(result),
        stringsAsFactors = FALSE
      )
      if (verbose) {
        cat("Start", start_id, "failed:", conditionMessage(result), "\n")
      }
      next
    }

    start_outcomes[[start_id]] <- data.frame(
      start = result$start,
      status = "completed",
      converged = result$converged,
      termination_reason = result$termination_reason,
      iter = result$iter,
      loglik = result$loglik,
      final_loglik_change = result$final_change,
      initial_jitter = result$initial_jitter,
      final_jitter = result$final_jitter,
      max_jitter = result$max_jitter,
      message = NA_character_,
      stringsAsFactors = FALSE
    )
    if (result$loglik > best_loglik) {
      best_loglik <- result$loglik
      best_result <- result
    }
  }

  start_outcomes <- do.call(rbind, start_outcomes)
  rownames(start_outcomes) <- NULL
  if (is.null(best_result)) {
    failures <- paste0(
      "start ", start_outcomes$start, " [",
      start_outcomes$termination_reason, "]: ", start_outcomes$message
    )
    stop(
      "All ", n_starts, " em_soft_lda starts failed: ",
      paste(failures, collapse = "; "),
      call. = FALSE
    )
  }

  returned_pi <- best_result$pi
  returned_mu <- sweep(best_result$mu, 2L, feature_center, FUN = "+")
  if (!is.null(colnames(PL))) {
    names(returned_pi) <- colnames(PL)
    rownames(returned_mu) <- colnames(PL)
  }
  colnames(returned_mu) <- colnames(X)
  returned_zeta <- best_result$state$zeta
  dimnames(returned_zeta) <- dimnames(PL)
  returned_covariance <- .disc_covariance_output(
    best_result$factor, dimension = d
  )
  class_mapping <- data.frame(
    pl_column = seq_len(K),
    class = if (is.null(colnames(PL))) {
      as.character(seq_len(K))
    } else {
      colnames(PL)
    },
    stringsAsFactors = FALSE
  )

  list(
    pi = returned_pi,
    mu = returned_mu,
    Sigma = returned_covariance,
    zeta = returned_zeta,
    loglik = best_result$loglik,
    iter = best_result$iter,
    diagnostics = list(
      converged = best_result$converged,
      termination_reason = best_result$termination_reason,
      final_loglik_change = best_result$final_change,
      loglik_history = best_result$history,
      chosen_start = best_result$start,
      starts = start_outcomes,
      covariance = list(
        policy = covariance,
        requested_ridge = reg,
        initial_jitter = best_result$initial_jitter,
        final_jitter = best_result$final_jitter,
        max_jitter = best_result$max_jitter,
        storage = if (covariance == "full") {
          "dense"
        } else {
          "sparse_diagonal"
        }
      ),
      class_mapping = class_mapping
    )
  )
}


#' @title Evidential-EM Logistic Regression with Soft Labels
#'
#' @description Fits multinomial logistic regression using E²M with soft labels,
#' a ridge penalty, and a monotone Newton line search.
#'
#' @param X Numeric matrix with observations in rows and features in columns.
#' @param PL Nonnegative numeric matrix with one plausibility row per
#'   observation and one column per class. Every row and column must have
#'   positive mass.
#' @param max_iter Positive integer maximum number of E²M iterations per start.
#' @param tol Positive relative convergence tolerance for the penalized
#'   evidential log-likelihood.
#' @param lambda Nonnegative ridge penalty applied to all coefficients.
#' @param n_starts Positive integer number of random initializations.
#' @param verbose Logical; print iteration diagnostics when `TRUE`.
#'
#' @return A list containing coefficient matrix `beta`, posterior memberships
#'   `zeta`, final unpenalized evidential log-likelihood `loglik`, and accepted
#'   E²M iteration count `iter`.
#'
#' @references
#' Quost, B., Denoeux, T., Li, S. (2017). Parametric classification with soft labels
#' using the Evidential EM algorithm. Advances in Data Analysis and Classification, 11(4), 659-690.
soft_lr <- function(X, PL, max_iter = 100, tol = 1e-6, lambda = 1e-5,
                    n_starts = 3, verbose = FALSE) {
  validated <- .disc_validate_evidential_inputs(X, PL)
  X <- validated$X
  PL <- validated$PL
  max_iter <- .disc_validate_positive_integer(max_iter, "max_iter")
  n_starts <- .disc_validate_positive_integer(n_starts, "n_starts")
  tol <- .disc_validate_scalar(tol, "tol", 0, strict = TRUE)
  lambda <- .disc_validate_scalar(lambda, "lambda", 0)
  verbose <- .disc_validate_verbose(verbose)

  n <- nrow(X)
  d <- ncol(X)
  K <- ncol(PL)
  q <- K - 1L
  Xtilde <- cbind(1, X)
  raw_log_PL <- log(PL)
  PL_log_offset <- apply(raw_log_PL, 1L, max)
  log_PL <- sweep(raw_log_PL, 1L, PL_log_offset, FUN = "-")
  loglik_offset <- sum(PL_log_offset)
  best_objective <- -Inf
  best_result <- NULL

  for (start_id in seq_len(n_starts)) {
    beta <- matrix(stats::rnorm((d + 1L) * q, sd = 0.1), d + 1L, q)
    current_state <- .disc_lr_e_step(Xtilde, beta, log_PL)
    current_objective <- current_state$loglik -
      0.5 * lambda * sum(beta^2)
    iteration_count <- 0L

    for (it in seq_len(max_iter)) {
      newton <- .disc_lr_newton_system(
        Xtilde, current_state, current_state$zeta, beta, lambda
      )
      gradient_tolerance <- tol * (1 + max(abs(beta)))
      if (max(abs(newton$gradient)) <= gradient_tolerance) {
        break
      }

      directional_derivative <- sum(newton$gradient * newton$direction)
      if (!is.finite(directional_derivative) ||
          directional_derivative <= 0) {
        if (verbose) {
          cat("Start", start_id, "stalled without an ascent direction\n")
        }
        break
      }

      current_q <- .disc_lr_q(
        current_state, current_state$zeta, beta, lambda
      )
      monotonic_tolerance <- 100 * .Machine$double.eps *
        (1 + abs(current_objective))
      accepted <- FALSE

      for (line_search in 0:50L) {
        step <- 0.5^line_search
        candidate_beta <- matrix(
          as.vector(beta) + step * newton$direction,
          nrow = d + 1L,
          ncol = q
        )
        candidate_model <- .disc_multinomial_state(
          Xtilde %*% candidate_beta
        )
        candidate_q <- .disc_lr_q(
          candidate_model, current_state$zeta, candidate_beta, lambda
        )
        normalized <- .disc_normalize_log_rows(
          log_PL + candidate_model$log_prob
        )
        candidate_loglik <- sum(normalized$log_norm)
        candidate_objective <- candidate_loglik -
          0.5 * lambda * sum(candidate_beta^2)
        armijo_target <- current_q +
          1e-4 * step * directional_derivative

        if (is.finite(candidate_q) && is.finite(candidate_objective) &&
            candidate_q >= armijo_target &&
            candidate_objective >=
              current_objective - monotonic_tolerance) {
          candidate_state <- c(
            candidate_model,
            list(zeta = normalized$weights, loglik = candidate_loglik)
          )
          accepted <- TRUE
          break
        }
      }

      if (!accepted) {
        if (verbose) {
          cat("Start", start_id,
              "stalled without a monotone Newton update\n")
        }
        break
      }

      previous_objective <- current_objective
      beta <- candidate_beta
      current_state <- candidate_state
      current_objective <- candidate_objective
      iteration_count <- it

      if (verbose) {
        cat("Start", start_id, "Iter", it, "LL:",
            current_state$loglik + loglik_offset, "Objective:",
            current_objective + loglik_offset, "\n")
      }
      if (abs(current_objective - previous_objective) <=
          tol * (1 + abs(previous_objective))) {
        break
      }
    }

    if (current_objective > best_objective) {
      coefficient_names <- colnames(X)
      if (is.null(coefficient_names)) {
        coefficient_names <- paste0("x", seq_len(d))
      }
      rownames(beta) <- c("(Intercept)", coefficient_names)
      if (!is.null(colnames(PL))) {
        colnames(beta) <- colnames(PL)[seq_len(q)]
      }
      dimnames(current_state$zeta) <- dimnames(PL)
      best_objective <- current_objective
      best_result <- list(
        beta = beta,
        zeta = current_state$zeta,
        loglik = current_state$loglik + loglik_offset,
        iter = iteration_count
      )
    }
  }

  best_result
}
