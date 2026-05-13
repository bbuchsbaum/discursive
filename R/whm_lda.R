#' Weighted Harmonic Mean of Trace Ratios (WHM-TR)
#'
#' Implements the Weighted Harmonic Mean of Trace Ratios method for multiclass
#' discriminant analysis (Li et al., 2017). Finds a linear transformation that
#' minimises the weighted harmonic mean of pairwise trace ratios derived from
#' between- and within-class scatter matrices.
#'
#' @param X A numeric matrix `n x d`.
#' @param y A factor or numeric vector of class labels of length `n`. If
#'   numeric, it is coerced to a factor.
#' @param ncomp Integer. Number of dimensions to retain.
#' @param preproc A preprocessing function from `multivarious`. Defaults to
#'   [multivarious::pass()] (no centering) for backward compatibility.
#'
#' @return A `discriminant_projector` (subclass `"whm_lda"`).
#'
#' @references
#' Li, Z. *et al.* (2017). "Beyond Trace Ratio: Weighted Harmonic Mean of
#' Trace Ratios for Multiclass Discriminant Analysis." *IEEE TKDE*.
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris[, 5]
#' proj <- whm_lda(X, y, ncomp = 2)
#' print(proj)
#' }
#'
#' @export
whm_lda <- function(X, y, ncomp, preproc = multivarious::pass()) {
  cl <- match.call()

  if (!requireNamespace("multivarious", quietly = TRUE)) {
    stop("Package 'multivarious' is required (for discriminant_projector) but not installed.")
  }

  if (!is.factor(y)) y <- as.factor(y)
  if (!is.numeric(ncomp) || length(ncomp) != 1 || ncomp < 1 || ncomp > ncol(X)) {
    stop("'ncomp' must be an integer between 1 and the number of features")
  }

  procres <- multivarious::prep(preproc)
  Xp <- multivarious::init_transform(procres, X)

  classes_y <- levels(y)
  c <- length(classes_y)

  initializeW <- function(d, m) {
    W <- matrix(rnorm(d * m), nrow = d, ncol = m)
    qr.Q(qr(W))
  }

  solveWHMTR <- function(X, y, m, max.iter = 100, tol = 1e-6) {
    d <- ncol(X)
    W <- initializeW(d, m)

    class_stats <- lapply(classes_y, function(cl_lab) {
      idx <- which(y == cl_lab)
      Xc  <- X[idx, , drop = FALSE]
      mu  <- colMeans(Xc)
      list(n  = nrow(Xc),
           mu = mu,
           Sw = crossprod(sweep(Xc, 2, mu)))
    })

    pair_stats <- list()
    for (j_idx in 1:(c - 1)) {
      for (k_idx in (j_idx + 1):c) {
        sj <- class_stats[[j_idx]]
        sk <- class_stats[[k_idx]]
        diff <- sj$mu - sk$mu
        pair_stats[[length(pair_stats) + 1]] <- list(
          Sw = sj$Sw + sk$Sw,
          Sb = tcrossprod(diff) * (sj$n + sk$n)
        )
      }
    }

    for (iter in seq_len(max.iter)) {
      M <- matrix(0, nrow = d, ncol = d)
      for (pair in pair_stats) {
        num_b <- sum(diag(crossprod(W, pair$Sb %*% W)))
        if (abs(num_b) < .Machine$double.eps) next
        num_w <- sum(diag(crossprod(W, pair$Sw %*% W)))
        M <- M + (1 / num_b) * pair$Sw - (num_w / (num_b^2)) * pair$Sb
      }

      eM <- eigen(M, symmetric = TRUE)
      idx_sorted <- order(eM$values, decreasing = FALSE)
      W_new <- eM$vectors[, idx_sorted[seq_len(m)], drop = FALSE]

      if (norm(W - W_new, type = "F") < tol) {
        W <- W_new
        break
      }
      W <- W_new
    }
    W
  }

  W_final <- solveWHMTR(Xp, y, ncomp)
  X_reduced <- Xp %*% W_final
  sdev_vec  <- apply(X_reduced, 2, sd)

  multivarious::discriminant_projector(
    v       = W_final,
    s       = X_reduced,
    sdev    = sdev_vec,
    preproc = procres,
    labels  = y,
    method  = "whm_lda",
    call    = cl,
    classes = "whm_lda"
  )
}
