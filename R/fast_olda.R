#' Compute Group Means
#'
#' Computes the mean of the rows of `X` for each level of the factor `y`.
#'
#' @param y A `factor` indicating group membership for each row of `X`.
#' @param X A `matrix` (`n x d`).
#' @return A matrix whose rows are the mean vectors for each group.
#' @export
group_means <- function(y, X) {

  if (!is.factor(y)) {
    stop("y must be a factor.")
  }

  if (all(table(y) == 1)) {
    rownames(X) <- names(table(y))
    return(X)
  } else {
    Rs <- rowsum(X, y)
    yt <- table(y)
    ret <- sweep(Rs, 1, yt, "/")
    row.names(ret) <- names(yt)
    return(ret)
  }
}

#' Compute Difference Matrix
#'
#' Helper used by [fastolda()] to build scatter matrices. Returns the
#' transposed successive differences of `X`.
#'
#' @param X A numeric matrix (`n x d`).
#' @return A `d x (n - 1)` matrix of successive differences.
#' @keywords internal
compute_Htdiff <- function(X) {
  if (!is.matrix(X)) {
    stop("X must be a matrix")
  }
  t(diff(X))
}

#' Fast Orthogonal LDA
#'
#' Performs a fast Orthogonal Linear Discriminant Analysis (OLDA) by computing
#' difference-based scatter matrices and solving for a discriminant projection.
#'
#' @param X A `matrix` (`n x d`).
#' @param y A `factor` (or coercible) of length `n` providing class labels.
#' @param ncomp Integer. Number of orthogonal discriminant directions to
#'   retain. If `NULL` (default), all directions returned by the algorithm are
#'   kept (at most `K - 1` for `K` classes).
#' @param preproc Preprocessing from `multivarious`. Defaults to
#'   [multivarious::center()].
#' @param reg Numeric regularisation parameter (default `0.01`).
#' @return A `discriminant_projector` object (subclass `"fastolda"`).
#' @export
fastolda <- function(X, y, ncomp = NULL,
                     preproc = multivarious::center(), reg = 0.01) {

  cl <- match.call()
  if (!is.factor(y)) y <- as.factor(y)

  procres <- multivarious::prep(preproc)
  X <- init_transform(procres, X)

  freq <- table(y)

  Ht_diff  <- compute_Htdiff(X)
  Ht_diffS <- crossprod(Ht_diff)
  m <- rowSums(Ht_diff) / nrow(X)

  HT <- sweep(Ht_diffS, 1, crossprod(m, Ht_diff), "-")
  HT <- cbind(t(-m %*% Ht_diff), HT)

  if (length(y) != ncol(HT)) {
    stop("Length of y must match the number of columns in HT")
  }

  ST <- tcrossprod(HT)
  ST <- ST + reg * diag(ncol(ST))

  RT <- chol(ST)
  RT_inv <- chol2inv(RT)

  HBs <- t(rowsum(t(HT), y))
  HB  <- sweep(HBs, 2, freq * sqrt(freq), "/")

  Hfinal <- Ht_diff %*% RT_inv %*% HB[, 1:(length(freq) - 1), drop = FALSE]

  GR <- qr(Hfinal)
  v  <- qr.Q(GR)

  if (!is.null(ncomp)) {
    ncomp <- as.integer(ncomp)
    if (ncomp < 1 || ncomp > ncol(v)) {
      stop(sprintf("'ncomp' must be between 1 and %d.", ncol(v)))
    }
    v <- v[, seq_len(ncomp), drop = FALSE]
  }

  s <- X %*% v

  multivarious::discriminant_projector(
    v       = v,
    s       = s,
    sdev    = apply(s, 2, sd),
    labels  = y,
    preproc = procres,
    method  = "fastolda",
    call    = cl,
    classes = "fastolda"
  )
}
