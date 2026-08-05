#' Experimental Multimodal Oriented Discriminant Analysis
#'
#' @description
#' Fits an experimental solver for the Multimodal Oriented Discriminant Analysis
#' (MODA) objective of De la Torre and Kanade (2005). The implementation:
#'
#' 1. **Clusters each class** into one or more clusters to capture multimodal structure.
#' 2. **Approximates each cluster's covariance** as \eqn{U_i \Lambda_i U_i^T + \sigma_i^2 I}
#'    to handle high-dimensional data (Section 6 of the paper).
#' 3. **Constructs the majorization function** \eqn{L(\mathbf{B})} that upper-bounds
#'    the Kullback–Leibler divergence-based objective \eqn{G(\mathbf{B})} (Equations (7)-(8)).
#' 4. **Iterates** along an Equation (11)-inspired direction, accepting a step
#'    only when the directly evaluated MODA objective decreases.
#'
#' This is not claimed to reproduce the paper's complete implementation: the
#' default clustering is k-means rather than multi-way normalized cuts, and the
#' step size is selected by backtracking on the MODA objective rather than the
#' paper's closed-form majorization step.
#'
#' @section References to the Paper:
#' - **Equation (7)**: Inequality used to construct the majorization function.
#' - **Equation (8)**: Definition of \eqn{L(\mathbf{B})} that majorizes \eqn{G(\mathbf{B})}.
#' - **Equation (9)**: Necessary condition for the minimum of \eqn{L(\mathbf{B})}.
#' - **Equation (10)**: Definition of \eqn{E_5(\mathbf{B})} to be minimized via gradient methods.
#' - **Equation (11)**: Normalized gradient-descent update with step size \eqn{\eta} chosen
#'   to minimize \eqn{E_5}.
#'
#' @param X A numeric matrix or data frame with observations in rows and
#'   features in columns, consistent with the other estimators in this package.
#' @param y A vector with one non-missing class label per row of `X` and at
#'   least two distinct classes.
#' @param k Integer. Dimensionality of the target subspace (number of features to extract).
#' @param numClusters Integer or vector/list specifying #clusters per class. If 1, it's ODA.
#' @param pcaFirst Logical. If TRUE, run PCA first to reduce dimension if \eqn{d >> n}. Defaults to TRUE.
#' @param pcaVar Fraction of variance to keep if `pcaFirst=TRUE`. Defaults to 0.95.
#' @param maxIter Maximum number of majorization iterations. Defaults to 50.
#' @param tol Convergence tolerance on relative change in the objective \eqn{G(\mathbf{B})}. Defaults to 1e-5.
#' @param clusterMethod Either `"kmeans"` or a custom function accepting (dataMatrix, kC) -> clusterIDs.
#' @param B_init Either `"random"` or `"pca"` to initialize the projection
#'   matrix \eqn{\mathbf{B}}.
#' @param verbose If TRUE, prints iteration progress.
#' @param lineSearchIter Number of line search iterations for step size selection (default = 20).
#' @param B_init_sd Standard deviation for the random initialization of
#'   \eqn{\mathbf{B}} if `B_init="random"`. Defaults to 1e-2.
#' @param preproc A fitted or unfitted preprocessing specification from
#'   \pkg{multivarious}. The default centers the predictors.
#' @param seed Optional nonnegative integer. When supplied, all stochastic
#'   clustering and initialization are run under a local seed and the caller's
#'   random-number state is restored on exit. With `NULL`, the caller's stream
#'   is used normally.
#'
#' @return A [multivarious::discriminant_projector()] with classes `"moda"`
#'   and `"moda_full"`. In addition to the standard projector fields, it
#'   contains `B` (an alias of the loadings), the accepted `objective_history`,
#'   `iterations`, `converged`, cluster assignments and fitted cluster models,
#'   PCA metadata, and initialization metadata.
#'
#' @details
#' **Key Steps**:
#'
#' 1. **Clustering** (Section 4): For each class, optionally split the samples into
#'    multiple clusters to model multimodality.
#' 2. **Approximate Covariances** (Section 6): For each cluster, approximate
#'    \eqn{\Sigma_i} by \eqn{\mathbf{U}_i \boldsymbol{\Lambda}_i
#'    \mathbf{U}_i^T + \sigma_i^2 \mathbf{I}}.
#' 3. **Majorization** (Sections 5.1–5.2): Build \eqn{L(\mathbf{B})} from \eqn{G(\mathbf{B})}
#'    using Equation (7) and sum up to get Equation (8).
#' 4. **Iterative Minimization** of \eqn{L(\mathbf{B}) \geq G(\mathbf{B})}. The partial
#'    derivatives (Equation (9)) yield a system of linear equations, solved here by
#'    gradient-based updates (Equations (10)–(11)).
#'
#' **High-Dimensional Data**: When \eqn{d \gg n}, it is recommended to set `pcaFirst=TRUE`
#' so that the dimension is reduced to at most \eqn{n}, avoiding rank deficiency and
#' improving generalization.
#'
#' **Classification after MODA**: Once \eqn{B} is learned, map a new sample \eqn{\mathbf{x}}
#' to \eqn{\mathbf{B}^T \mathbf{x}} (plus PCA if used) and classify in that lower-dimensional space.
#'
#' For further details, see:
#' \enumerate{
#'   \item De la Torre & Kanade (2005). "Multimodal Oriented Discriminant Analysis."
#'   \item Equations (7)–(11) for the majorization steps.
#'   \item Section 6 for the covariance factorization in high dimensions.
#' }
#'
#' @examples
#' # Synthetic example (small scale):
#' set.seed(123)
#' d <- 20; n <- 40
#' X <- matrix(rnorm(d*n), nrow = n, ncol = d)
#' y <- rep(1:2, each = n/2)
#' res <- moda_full(X, y, k = 2, numClusters = 1, pcaFirst = FALSE,
#'                  B_init = "pca", maxIter = 15)
#'
#' # Inspect the learned projection B
#' str(res)
#'
#' @export
moda_full <- function(X,
                      y,
                      k,
                      numClusters    = 1,
                      pcaFirst       = TRUE,
                      pcaVar         = 0.95,
                      maxIter        = 50,
                      tol            = 1e-5,
                      clusterMethod  = "kmeans",
                      B_init         = "random",
                      verbose        = FALSE,
                      lineSearchIter = 20,
                      B_init_sd      = 1e-2,
                      preproc        = center(),
                      seed           = NULL) {
  X <- as.matrix(X)
  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) < 2L || ncol(X) < 1L) {
    stop("'X' must be a numeric matrix with at least two rows and one column.")
  }
  if (any(!is.finite(X))) {
    stop("'X' must contain only finite values.")
  }

  n <- nrow(X)
  d <- ncol(X)
  if (length(y) != n) {
    stop("'y' must contain one label per row of 'X'.")
  }
  if (anyNA(y)) {
    stop("'y' must not contain missing labels.")
  }
  y <- droplevels(factor(y))
  uniqueLabels <- levels(y)
  if (length(uniqueLabels) < 2) {
    stop("'y' must contain at least two distinct classes.")
  }
  if (!is.logical(pcaFirst) || length(pcaFirst) != 1L || is.na(pcaFirst)) {
    stop("'pcaFirst' must be TRUE or FALSE.")
  }
  pcaVar <- .check_finite_scalar(
    pcaVar, "pcaVar", lower = 0, upper = 1, lower_open = TRUE
  )
  k <- .check_whole_number(k, "k", lower = 1L, upper = d)
  maxIter <- .check_whole_number(maxIter, "maxIter", lower = 1L)
  lineSearchIter <- .check_whole_number(
    lineSearchIter, "lineSearchIter", lower = 1L
  )
  tol <- .check_finite_scalar(tol, "tol", lower = 0, lower_open = TRUE)
  B_init_sd <- .check_finite_scalar(
    B_init_sd, "B_init_sd", lower = 0, lower_open = TRUE
  )
  B_init <- match.arg(B_init, c("random", "pca"))
  if (!is.null(seed)) {
    seed <- .check_whole_number(
      seed, "seed", lower = 0L, upper = .Machine$integer.max
    )
    had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (had_seed) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    }
    on.exit({
      if (had_seed) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed)
  }

  class_sizes <- tabulate(y, nbins = nlevels(y))
  if (length(numClusters) == 1L) {
    numClusters <- rep(numClusters, nlevels(y))
  }
  if (length(numClusters) != nlevels(y)) {
    stop("'numClusters' must have length one or one value per class.")
  }
  numClusters <- vapply(seq_along(numClusters), function(i) {
    .check_whole_number(
      numClusters[[i]], "numClusters", lower = 1L, upper = class_sizes[i]
    )
  }, integer(1L))

  procres <- multivarious::fit(preproc, X)
  Xp <- as.matrix(multivarious::transform(procres, X))
  if (!is.numeric(Xp) || any(dim(Xp) != dim(X)) || any(!is.finite(Xp))) {
    stop("'preproc' must produce a finite numeric matrix with the same dimensions as 'X'.")
  }
  # The paper's algebra stores samples in columns; keep that orientation only
  # inside the solver while exposing the package-wide row-sample API.
  X_internal <- t(Xp)

  # ------------------
  # 1) PCA if requested
  # ------------------
  pcaInfo <- NULL
  X_work  <- X_internal

  if (pcaFirst) {
    # Always do PCA, regardless of d vs n
    pcaInfo <- .pca_reduce_data(X_internal, varFraction = pcaVar)
    X_work  <- t(pcaInfo$U) %*% sweep(X_internal, 1, pcaInfo$mean, "-")
    dNew    <- nrow(X_work)

    # Now check if k <= dNew
    if (k > dNew) {
      stop("'k' cannot exceed the dimension after PCA. ",
           "Try reducing 'k' or adjusting 'pcaVar'.")
    }
  } else {
    # No PCA
    dNew <- d
    if (k > dNew) {
      stop("'k' cannot exceed the original dimension 'd' in no-PCA mode.")
    }
  }

  # ------------------
  # 2) Cluster each class -> for MODA. If numClusters=1 => ODA
  # ------------------
  cluster_assignments <- .cluster_each_class(X_work, y, numClusters, clusterMethod)

  ## ------------------
  ## 3) Covariance Approx.
  ##    For each class & cluster: Sigma ~ U_i Lambda_i U_i^T + sigma^2 I
  ## ------------------
  classes  <- uniqueLabels
  nClasses <- length(classes)
  SigmaApprox <- vector("list", nClasses)
  MuApprox    <- vector("list", nClasses)
  for (iC in seq_along(classes)) {
    labelC   <- classes[iC]
    idxC     <- which(y == labelC)
    cAssign  <- cluster_assignments[[iC]]
    kC       <- max(cAssign)
    SigmaApprox[[iC]] <- vector("list", kC)
    MuApprox[[iC]]    <- vector("list", kC)

    for (clu in seq_len(kC)) {
      idxCluster <- (cAssign == clu)
      Xc         <- X_work[, idxC[idxCluster], drop = FALSE]
      covApprox  <- .approximate_covariance(Xc) # from Section 6
      SigmaApprox[[iC]][[clu]] <- covApprox
      MuApprox[[iC]][[clu]]    <- rowMeans(Xc)
    }
  }

  ## ------------------
  ## 4) Initialize B
  ## ------------------
  if (B_init == "random") {
    B <- matrix(rnorm(dNew * k, sd = B_init_sd), nrow = dNew, ncol = k)
  } else if (B_init == "pca") {
    # PCA-based initialization within the (possibly reduced) space X_work
    pca_b_init <- .pca_reduce_data(X_work, varFraction = 1.0)
    if (ncol(pca_b_init$U) < k) {
      stop("PCA init failed: fewer components than 'k'. Decrease 'k' or use 'random'.")
    }
    B <- pca_b_init$U[, seq_len(k), drop = FALSE]
  } else {
    stop("'B_init' must be 'random' or 'pca'.")
  }

  ## ------------------
  ## 5) Objective G(B) for monitoring
  ##    G(B) = - sum_{i,r1} sum_{j!=i, r2} KL(...) (Eqn (5)-(6))
  ## ------------------
  G_of_B <- function(Bmat) {
    totalVal <- 0
    # Loop over classes/clusters
    for (iC in seq_along(classes)) {
      kC1 <- length(SigmaApprox[[iC]])
      for (r1 in seq_len(kC1)) {
        # (B^T Sigma_i^r1 B)^-1
        SB       <- .multiply_sigma_B(SigmaApprox[[iC]][[r1]], Bmat)
        BtSB     <- crossprod(Bmat, SB)
        invBtSB  <- .safe_invert(BtSB)

        mu_i_r1  <- MuApprox[[iC]][[r1]]
        sumTr    <- 0
        # Summation over other classes/clusters
        for (jC in seq_along(classes)) {
          if (jC == iC) next
          kC2 <- length(SigmaApprox[[jC]])
          for (r2 in seq_len(kC2)) {
            mu_j_r2   <- MuApprox[[jC]][[r2]]
            dm        <- mu_i_r1 - mu_j_r2
            Btdm      <- crossprod(Bmat, dm)
            sumTr     <- sumTr + as.numeric(t(Btdm) %*% invBtSB %*% Btdm)

            # Cov part
            SB_jr2    <- .multiply_sigma_B(SigmaApprox[[jC]][[r2]], Bmat)
            BtSB_jr2  <- crossprod(Bmat, SB_jr2)
            sumTr     <- sumTr + sum(invBtSB * BtSB_jr2)
          }
        }
        totalVal <- totalVal + sumTr
      }
    }
    return(-totalVal)
  }

  oldObj <- G_of_B(B)
  objective_history <- numeric(maxIter + 1L)
  objective_history[1L] <- oldObj
  converged <- FALSE

  ## ------------------
  ## 6) Iterative Majorization (Eqn (7)-(11))
  ## ------------------
  for (iter in seq_len(maxIter)) {
    T_sum <- matrix(0, nrow = dNew, ncol = k)
    S_sum <- matrix(0, nrow = dNew, ncol = k)

    # Build partial sums for eqn. (9): T_i, F_i => then eqn. (10)-(11)
    for (iC in seq_along(classes)) {
      kC1 <- length(SigmaApprox[[iC]])
      for (r1 in seq_len(kC1)) {
        Sig_i_r1 <- SigmaApprox[[iC]][[r1]]
        SB       <- .multiply_sigma_B(Sig_i_r1, B)
        BtSB     <- crossprod(B, SB)
        invBtSB  <- .safe_invert(BtSB)

        # B^T A_i^{r1} B
        BtAB     <- matrix(0, nrow = k, ncol = k)
        mu_i_r1  <- MuApprox[[iC]][[r1]]
        for (jC in seq_along(classes)) {
          if (jC == iC) next
          kC2 <- length(SigmaApprox[[jC]])
          for (r2 in seq_len(kC2)) {
            mu_j_r2  <- MuApprox[[jC]][[r2]]
            dm       <- mu_i_r1 - mu_j_r2
            Btdm     <- crossprod(B, dm)
            BtAB     <- BtAB + (Btdm %*% t(Btdm))

            # Cov part
            SB_jr2   <- .multiply_sigma_B(SigmaApprox[[jC]][[r2]], B)
            BtSB_jr2 <- crossprod(B, SB_jr2)
            BtAB     <- BtAB + BtSB_jr2
          }
        }
        # F_i
        Fi  <- invBtSB %*% BtAB %*% invBtSB

        # T_i
        Ti_r1 <- matrix(0, nrow = dNew, ncol = k)
        for (jC in seq_along(classes)) {
          if (jC == iC) next
          kC2 <- length(SigmaApprox[[jC]])
          for (r2 in seq_len(kC2)) {
            mu_j_r2 <- MuApprox[[jC]][[r2]]
            dm      <- mu_i_r1 - mu_j_r2
            dmMat   <- dm %*% t(dm)
            tmp_dm  <- dmMat %*% B
            tmp_cov <- .multiply_sigma_B(SigmaApprox[[jC]][[r2]], B)
            tmp_all <- tmp_dm + tmp_cov
            tmp_fin <- tmp_all %*% invBtSB
            Ti_r1   <- Ti_r1 + tmp_fin
          }
        }

        # Accumulate
        T_sum  <- T_sum + Ti_r1
        SigmaBF<- .multiply_sigma_B(Sig_i_r1, B) %*% Fi
        S_sum  <- S_sum + SigmaBF
      }
    }

    Rk <- T_sum - S_sum
    currObjVal <- oldObj
    bestObjVal <- currObjVal
    alpha      <- 1.0

    # Basic line search
    E5_of_B <- function(Bcandidate) G_of_B(Bcandidate)

    for (lsIter in seq_len(lineSearchIter)) {
      B_try     <- B + alpha * Rk
      newObjVal <- E5_of_B(B_try)
      if (newObjVal < bestObjVal) {
        B         <- B_try
        bestObjVal<- newObjVal
        break
      } else {
        alpha <- alpha / 2
      }
    }

    newObj      <- G_of_B(B)
    objective_history[iter + 1L] <- newObj
    if (verbose) {
      cat(sprintf("[Iter %d] oldObj=%.6f, newObj=%.6f, alpha=%.3g\n",
                  iter, currObjVal, newObj, alpha))
    }

    # Convergence check
    relChange <- abs(newObj - oldObj) / (abs(oldObj) + 1e-12)
    if (relChange < tol) {
      if (verbose) cat("Converged.\n")
      converged <- TRUE
      break
    }
    oldObj <- newObj
  }
  objective_history <- objective_history[seq_len(iter + 1L)]

  ## ------------------
  ## 7) Expand B if PCA used
  ## ------------------
  if (!is.null(pcaInfo)) {
    B <- pcaInfo$U %*% B
  }

  decomposition <- qr(B, tol = tol)
  if (decomposition$rank < k) {
    stop("The fitted MODA basis is numerically rank deficient.")
  }
  vectors <- qr.Q(decomposition)[, seq_len(k), drop = FALSE]
  signs <- vapply(seq_len(k), function(j) {
    sign(vectors[which.max(abs(vectors[, j])), j])
  }, numeric(1L))
  signs[signs == 0] <- 1
  vectors <- sweep(vectors, 2L, signs, FUN = "*")
  component_names <- paste0("MODA", seq_len(k))
  rownames(vectors) <- colnames(X)
  colnames(vectors) <- component_names
  scores <- Xp %*% vectors
  colnames(scores) <- component_names
  score_sdev <- apply(scores, 2L, stats::sd)
  names(score_sdev) <- component_names

  multivarious::discriminant_projector(
    v = vectors,
    s = scores,
    sdev = score_sdev,
    preproc = procres,
    labels = y,
    classes = c("moda", "moda_full"),
    B = vectors,
    objVals = objective_history[-1L],
    objective_history = objective_history,
    iterations = iter,
    converged = converged,
    clusters = cluster_assignments,
    cluster_covariances = SigmaApprox,
    cluster_means = MuApprox,
    pcaInfo = pcaInfo,
    pca_first = pcaFirst,
    num_clusters = numClusters,
    initialization = B_init,
    seed = seed,
    algorithm = "guarded_objective_descent"
  )
}


# ==============================================================================
# Internal Supporting Functions
# ==============================================================================

#' PCA-based Dimensionality Reduction
#' @keywords internal
.pca_reduce_data <- function(X, varFraction = 0.95) {
  # X: d x n
  mu <- rowMeans(X)
  Xc <- sweep(X, 1, mu, "-")
  svdOut <- svd(Xc, nu=min(nrow(Xc),ncol(Xc)), nv=min(nrow(Xc),ncol(Xc)))
  dVals  <- svdOut$d
  cumVar <- cumsum(dVals^2) / sum(dVals^2 + 1e-15)
  r <- which(cumVar >= varFraction)[1]
  if (is.na(r)) r <- length(dVals)
  Ukeep <- svdOut$u[, seq_len(r), drop=FALSE]
  list(U=Ukeep, mean=mu)
}

#' Cluster each class into multiple clusters
#' @keywords internal
.cluster_each_class <- function(X_work, y, numClusters, method) {
  classes <- levels(droplevels(factor(y)))
  clusterList <- vector("list", length(classes))
  for (i in seq_along(classes)) {
    labelC <- classes[i]
    idxC   <- which(y == labelC)
    Xc     <- X_work[, idxC, drop=FALSE]

    # Determine #clusters for this class
    if (length(numClusters) == 1) {
      kC <- numClusters
    } else {
      kC <- numClusters[i]
    }
    if (kC < 1) {
      stop("numClusters must be >= 1.")
    }

    if (kC == 1) {
      clusterList[[i]] <- rep(1, length(idxC))
    } else if (identical(method, "kmeans")) {
      # kmeans with 5 restarts
      km <- tryCatch({
        kmeans(t(Xc), centers=kC, nstart=5)
      }, error=function(e){
        warning(sprintf(
          "kmeans for class '%s' failed: %s. Reverting to single cluster.",
          as.character(labelC), e$message
        ))
        NULL
      })
      if (is.null(km)) {
        clusterList[[i]] <- rep(1, length(idxC))
      } else {
        clusterList[[i]] <- km$cluster
      }
    } else if (is.function(method)) {
      userClust <- method(Xc, kC)
      if (!is.numeric(userClust) || length(userClust) != ncol(Xc)) {
        stop("Custom cluster function must return integer clusterIDs for each sample.")
      }
      clusterList[[i]] <- userClust
    } else {
      stop("clusterMethod must be 'kmeans' or a function(Xmat, k).")
    }
  }
  clusterList
}

#' Approximate Covariance as U_i Lambda_i U_i^T + sigma^2 I  (Section 6)
#' @keywords internal
.approximate_covariance <- function(Xc, energyThreshold=0.90) {
  # Xc: dNew x nC
  d2 <- nrow(Xc)
  nC <- ncol(Xc)
  if (nC <= 1) {
    # If only 1 sample, trivial covariance
    return(list(
      U       = diag(d2),
      Lambda  = matrix(0, d2, d2),
      sigma2  = 1e-8,
      mean    = if (nC == 1) rowMeans(Xc) else rep(0, d2)
    ))
  }
  mu   <- rowMeans(Xc)
  Xctr <- sweep(Xc, 1, mu, "-")
  S    <- (1/(nC-1)) * (Xctr %*% t(Xctr))
  eigS <- eigen(S, symmetric = TRUE)
  vals <- pmax(eigS$values, 0)
  vecs <- eigS$vectors

  cumE <- cumsum(vals) / sum(vals + 1e-15)
  l    <- which(cumE >= energyThreshold)[1]
  if (is.na(l)) l <- length(vals)

  leftover <- if (l < length(vals)) vals[(l + 1L):length(vals)] else numeric()
  covariance_scale <- max(1, mean(vals))
  sigma2 <- if (length(leftover) > 0L) mean(leftover) else 0
  sigma2 <- max(sigma2, sqrt(.Machine$double.eps) * covariance_scale)

  lam  <- vals[seq_len(l)] - sigma2
  lam[lam < 0] <- 0  # numerical safeguard
  U_i  <- vecs[, seq_len(l), drop=FALSE]
  list(
    U      = U_i,
    Lambda = diag(lam, nrow=l),
    sigma2 = sigma2,
    mean   = mu
  )
}

#' Multiply approximate covariance by B
#' Sigma_i = U_i Lambda_i U_i^T + sigma^2 I
#' => Sigma_i B = U_i Lambda_i (U_i^T B) + sigma^2 B
#' @keywords internal
.multiply_sigma_B <- function(SigmaApprox, B) {
  U       <- SigmaApprox$U
  Lambda  <- SigmaApprox$Lambda
  sigma2  <- SigmaApprox$sigma2
  tmp     <- crossprod(U, B)         # l x k
  tmp2    <- Lambda %*% tmp          # l x k
  out     <- U %*% tmp2 + sigma2 * B # dNew x k
  out
}

#' Safely invert a matrix, fallback to pseudo-inverse if needed
#' @keywords internal
.safe_invert <- function(M) {
  tryCatch(
    solve(M),
    error = function(e) {
      # fallback
      MASS::ginv(M)
    }
  )
}
