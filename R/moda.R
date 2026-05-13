#' Multimodal Oriented Discriminant Analysis (MODA)
#'
#' Implements the Multimodal Oriented Discriminant Analysis framework of
#' De la Torre & Kanade (2005). The method clusters each class into one or
#' more modes, approximates each cluster's covariance as
#' `U_i Lambda_i U_i^T + sigma^2 I`, builds a majorisation function
#' `L(B)` upper-bounding the KL-divergence-based objective `G(B)`, and
#' iteratively minimises `L(B)` via a normalised gradient-descent update with
#' line search.
#'
#' @param X A numeric matrix of size `n x d` (rows = samples, columns =
#'   features). This is the standard `discursive` orientation. The matrix is
#'   transposed internally to match the paper's `d x n` convention.
#' @param y A vector of length `n` of class labels (`>= 2` distinct labels).
#' @param ncomp Integer. Dimensionality of the target subspace. (Was `k` in
#'   earlier versions.)
#' @param numClusters Integer or vector specifying number of clusters per
#'   class. If `1`, reduces to ODA.
#' @param pcaFirst Logical — run PCA first to reduce dimension when
#'   `d >> n`. Default `TRUE`.
#' @param pcaVar Fraction of variance to keep when `pcaFirst = TRUE`. Default
#'   `0.95`.
#' @param maxIter Maximum majorisation iterations. Default `50`.
#' @param tol Convergence tolerance on the relative change in `G(B)`. Default
#'   `1e-5`.
#' @param clusterMethod Either `"kmeans"` or a function `(dataMatrix, kC) ->
#'   clusterIDs`.
#' @param B_init Either `"random"` or `"pca"` to initialise the projection
#'   matrix `B`.
#' @param verbose If `TRUE`, prints iteration progress.
#' @param lineSearchIter Maximum line-search iterations (default `20`).
#' @param B_init_sd Standard deviation for random initialisation of `B`.
#'   Default `1e-2`.
#'
#' @return A `discriminant_projector` (subclass `"moda_full"`) with extra
#'   fields `objVals`, `clusters`, and `pcaInfo`.
#'
#' @references
#' De la Torre, F., & Kanade, T. (2005). "Multimodal Oriented Discriminant
#' Analysis."
#'
#' @examples
#' set.seed(123)
#' n <- 40; d <- 20
#' X <- matrix(rnorm(n*d), nrow = n, ncol = d)
#' y <- rep(1:2, each = n/2)
#' fit <- moda_full(X, y, ncomp = 2, numClusters = 1,
#'                  pcaFirst = FALSE, maxIter = 15)
#' print(fit)
#'
#' @export
moda_full <- function(X, y,
                      ncomp,
                      numClusters    = 1,
                      pcaFirst       = TRUE,
                      pcaVar         = 0.95,
                      maxIter        = 50,
                      tol            = 1e-5,
                      clusterMethod  = "kmeans",
                      B_init         = "random",
                      verbose        = FALSE,
                      lineSearchIter = 20,
                      B_init_sd      = 1e-2) {

  cl <- match.call()
  k <- ncomp

  if (!is.matrix(X)) X <- as.matrix(X)
  X_in <- X            # n x d, kept for projecting later
  Xmat <- t(X)         # d x n internally to match paper notation

  d <- nrow(Xmat)
  n <- ncol(Xmat)
  if (length(y) != n) {
    stop("Length of 'y' must match the number of rows in X.")
  }
  uniqueLabels <- unique(y)
  if (length(uniqueLabels) < 2) {
    stop("There must be at least 2 distinct classes in 'y'.")
  }
  if (!is.logical(pcaFirst)) {
    stop("'pcaFirst' must be TRUE or FALSE.")
  }
  if (pcaVar <= 0 || pcaVar > 1) {
    stop("'pcaVar' must be in the range (0, 1].")
  }

  pcaInfo <- NULL
  X_work  <- Xmat

  if (pcaFirst) {
    pcaInfo <- .pca_reduce_data(Xmat, varFraction = pcaVar)
    X_work  <- t(pcaInfo$U) %*% sweep(Xmat, 1, pcaInfo$mean, "-")
    dNew    <- nrow(X_work)
    if (k > dNew) {
      stop("'ncomp' cannot exceed the dimension after PCA. ",
           "Try reducing 'ncomp' or adjusting 'pcaVar'.")
    }
  } else {
    dNew <- d
    if (k > dNew) {
      stop("'ncomp' cannot exceed the original feature dimension when pcaFirst = FALSE.")
    }
  }

  cluster_assignments <- .cluster_each_class(X_work, y, numClusters, clusterMethod)

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
      Xc <- X_work[, idxC[idxCluster], drop = FALSE]
      SigmaApprox[[iC]][[clu]] <- .approximate_covariance(Xc)
      MuApprox[[iC]][[clu]]    <- rowMeans(Xc)
    }
  }

  set.seed(1)
  if (B_init == "random") {
    B <- matrix(rnorm(dNew * k, sd = B_init_sd), nrow = dNew, ncol = k)
  } else if (B_init == "pca") {
    pca_b_init <- .pca_reduce_data(X_work, varFraction = 1.0)
    if (ncol(pca_b_init$U) < k) {
      stop("PCA init failed: fewer components than 'ncomp'. Decrease 'ncomp' or use 'random'.")
    }
    B <- pca_b_init$U[, seq_len(k), drop = FALSE]
  } else {
    stop("'B_init' must be 'random' or 'pca'.")
  }

  G_of_B <- function(Bmat) {
    totalVal <- 0
    for (iC in seq_along(classes)) {
      kC1 <- length(SigmaApprox[[iC]])
      for (r1 in seq_len(kC1)) {
        SB       <- .multiply_sigma_B(SigmaApprox[[iC]][[r1]], Bmat)
        BtSB     <- crossprod(Bmat, SB)
        invBtSB  <- .safe_invert(BtSB)
        mu_i_r1  <- MuApprox[[iC]][[r1]]
        sumTr    <- 0
        for (jC in seq_along(classes)) {
          if (jC == iC) next
          kC2 <- length(SigmaApprox[[jC]])
          for (r2 in seq_len(kC2)) {
            mu_j_r2 <- MuApprox[[jC]][[r2]]
            dm      <- mu_i_r1 - mu_j_r2
            Btdm    <- crossprod(Bmat, dm)
            sumTr   <- sumTr + as.numeric(t(Btdm) %*% invBtSB %*% Btdm)
            SB_jr2  <- .multiply_sigma_B(SigmaApprox[[jC]][[r2]], Bmat)
            BtSB_jr2 <- crossprod(Bmat, SB_jr2)
            sumTr   <- sumTr + sum(invBtSB * BtSB_jr2)
          }
        }
        totalVal <- totalVal + sumTr
      }
    }
    -totalVal
  }

  oldObj  <- G_of_B(B)
  objVals <- numeric(maxIter)

  for (iter in seq_len(maxIter)) {
    T_sum <- matrix(0, nrow = dNew, ncol = k)
    S_sum <- matrix(0, nrow = dNew, ncol = k)

    for (iC in seq_along(classes)) {
      kC1 <- length(SigmaApprox[[iC]])
      for (r1 in seq_len(kC1)) {
        Sig_i_r1 <- SigmaApprox[[iC]][[r1]]
        SB       <- .multiply_sigma_B(Sig_i_r1, B)
        BtSB     <- crossprod(B, SB)
        invBtSB  <- .safe_invert(BtSB)

        BtAB     <- matrix(0, nrow = k, ncol = k)
        mu_i_r1  <- MuApprox[[iC]][[r1]]
        for (jC in seq_along(classes)) {
          if (jC == iC) next
          kC2 <- length(SigmaApprox[[jC]])
          for (r2 in seq_len(kC2)) {
            mu_j_r2 <- MuApprox[[jC]][[r2]]
            dm      <- mu_i_r1 - mu_j_r2
            Btdm    <- crossprod(B, dm)
            BtAB    <- BtAB + (Btdm %*% t(Btdm))
            SB_jr2  <- .multiply_sigma_B(SigmaApprox[[jC]][[r2]], B)
            BtSB_jr2 <- crossprod(B, SB_jr2)
            BtAB    <- BtAB + BtSB_jr2
          }
        }
        Fi <- invBtSB %*% BtAB %*% invBtSB

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
            tmp_fin <- (tmp_dm + tmp_cov) %*% invBtSB
            Ti_r1   <- Ti_r1 + tmp_fin
          }
        }

        T_sum <- T_sum + Ti_r1
        S_sum <- S_sum + .multiply_sigma_B(Sig_i_r1, B) %*% Fi
      }
    }

    Rk <- T_sum - S_sum
    currObjVal <- oldObj
    bestObjVal <- currObjVal
    alpha      <- 1.0

    E5_of_B <- function(Bcandidate) G_of_B(Bcandidate)

    for (lsIter in seq_len(lineSearchIter)) {
      B_try     <- B + alpha * Rk
      newObjVal <- E5_of_B(B_try)
      if (newObjVal < bestObjVal) {
        B          <- B_try
        bestObjVal <- newObjVal
        break
      } else {
        alpha <- alpha / 2
      }
    }

    newObj         <- G_of_B(B)
    objVals[iter]  <- newObj
    if (verbose) {
      cat(sprintf("[Iter %d] oldObj=%.6f, newObj=%.6f, alpha=%.3g\n",
                  iter, currObjVal, newObj, alpha))
    }

    relChange <- abs(newObj - oldObj) / (abs(oldObj) + 1e-12)
    if (relChange < tol) {
      if (verbose) cat("Converged.\n")
      objVals <- objVals[seq_len(iter)]
      break
    }
    oldObj <- newObj
  }

  if (!is.null(pcaInfo)) {
    B <- pcaInfo$U %*% B
  }

  s_mat <- X_in %*% B
  multivarious::discriminant_projector(
    v       = B,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    labels  = y,
    objVals = objVals,
    clusters = cluster_assignments,
    pcaInfo = pcaInfo,
    method  = "moda_full",
    call    = cl,
    classes = "moda_full"
  )
}


# ==============================================================================
# Internal Supporting Functions
# ==============================================================================

#' @keywords internal
.pca_reduce_data <- function(X, varFraction = 0.95) {
  mu <- rowMeans(X)
  Xc <- sweep(X, 1, mu, "-")
  svdOut <- svd(Xc, nu = min(nrow(Xc), ncol(Xc)), nv = min(nrow(Xc), ncol(Xc)))
  dVals  <- svdOut$d
  cumVar <- cumsum(dVals^2) / sum(dVals^2 + 1e-15)
  r <- which(cumVar >= varFraction)[1]
  if (is.na(r)) r <- length(dVals)
  list(U = svdOut$u[, seq_len(r), drop = FALSE], mean = mu)
}

#' @keywords internal
.cluster_each_class <- function(X_work, y, numClusters, method) {
  classes <- unique(y)
  clusterList <- vector("list", length(classes))
  for (i in seq_along(classes)) {
    labelC <- classes[i]
    idxC   <- which(y == labelC)
    Xc     <- X_work[, idxC, drop = FALSE]

    kC <- if (length(numClusters) == 1) numClusters else numClusters[i]
    if (kC < 1) stop("numClusters must be >= 1.")

    if (kC == 1) {
      clusterList[[i]] <- rep(1, length(idxC))
    } else if (identical(method, "kmeans")) {
      km <- tryCatch(kmeans(t(Xc), centers = kC, nstart = 5),
                     error = function(e) {
                       warning(sprintf(
                         "kmeans for class '%s' failed: %s. Reverting to single cluster.",
                         as.character(labelC), e$message
                       ))
                       NULL
                     })
      clusterList[[i]] <- if (is.null(km)) rep(1, length(idxC)) else km$cluster
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

#' @keywords internal
.approximate_covariance <- function(Xc, energyThreshold = 0.90) {
  d2 <- nrow(Xc)
  nC <- ncol(Xc)
  if (nC <= 1) {
    return(list(
      U       = diag(d2),
      Lambda  = matrix(0, d2, d2),
      sigma2  = 1e-8,
      mean    = if (nC == 1) rowMeans(Xc) else rep(0, d2)
    ))
  }
  mu   <- rowMeans(Xc)
  Xctr <- sweep(Xc, 1, mu, "-")
  S    <- (1 / (nC - 1)) * (Xctr %*% t(Xctr))
  eigS <- eigen(S, symmetric = TRUE)
  vals <- eigS$values
  vecs <- eigS$vectors

  cumE <- cumsum(vals) / sum(vals + 1e-15)
  l    <- which(cumE >= energyThreshold)[1]
  if (is.na(l)) l <- length(vals)

  leftover <- vals[(l + 1):length(vals)]
  sigma2   <- if (length(leftover) > 0) mean(leftover) else 0

  lam <- vals[seq_len(l)] - sigma2
  lam[lam < 0] <- 0
  list(U = vecs[, seq_len(l), drop = FALSE],
       Lambda = diag(lam, nrow = l),
       sigma2 = sigma2,
       mean = mu)
}

#' @keywords internal
.multiply_sigma_B <- function(SigmaApprox, B) {
  U      <- SigmaApprox$U
  Lambda <- SigmaApprox$Lambda
  sigma2 <- SigmaApprox$sigma2
  tmp    <- crossprod(U, B)
  tmp2   <- Lambda %*% tmp
  U %*% tmp2 + sigma2 * B
}

#' @keywords internal
.safe_invert <- function(M) {
  tryCatch(solve(M), error = function(e) MASS::ginv(M))
}
