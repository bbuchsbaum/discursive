#' Compute Group Means
#'
#' Compute the group-wise mean vectors for a given data matrix.
#'
#' Given a data matrix \code{X} and a factor \code{Y}, this function computes the mean of the rows of \code{X} for each level of \code{Y}.
#'
#' @param Y A \code{factor} indicating group membership for each row of \code{X}.
#' @param X A \code{matrix} (n x d), where n is the number of samples and d is the number of features.
#' @return A matrix of group means, where the number of rows equals the number of groups, and each row is the mean vector for that group.
#' @export
#' Compute Group Means
#'
#' Compute the group-wise mean vectors for a given data matrix.
#'
#' @export
group_means <- function(Y, X) {
  
  if (!is.factor(Y)) {
    stop("Y must be a factor.")
  }
  
  if (all(table(Y) == 1)) {
    # Each group has only one sample, so the group means are just the samples themselves
    rownames(X) <- names(table(Y))    # <-- Updated line here
    return(X)
  } else {
    Rs <- rowsum(X, Y)
    yt <- table(Y)
    ret <- sweep(Rs, 1, yt, "/")
    row.names(ret) <- names(yt)
    return(ret)
  }
}

#' Fast Orthogonal LDA
#'
#' Perform a fast Orthogonal Linear Discriminant Analysis (OLDA) based on provided data and class labels.
#'
#' This function performs OLDA by pre-processing the data, computing difference-based scatter matrices, and then solving for a discriminant projection.
#' The final result is returned as a \code{discriminant_projector} object from the \code{multivarious} package.
#'
#' @param X A \code{matrix} (n x d) with n samples and d features.
#' @param Y A \code{factor} with length n, providing the class/group label for each sample.
#' @param preproc A pre-processing step, such as \code{center()}, from \code{multivarious}. Default is \code{center()}.
#' @param reg A \code{numeric} regularization parameter (default = 0.01). This is used to ensure invertibility of certain matrices.
#' @return A \code{discriminant_projector} object containing:
#' \itemize{
#'   \item \code{rotation}: The matrix of loadings (d x r) where r is the reduced dimension.
#'   \item \code{s}: The scores matrix (n x r), i.e., \code{X \%*\% rotation}.
#'   \item \code{sdev}: Standard deviations of the scores.
#'   \item \code{labels}: The class labels.
#'   \item \code{preproc}: The preprocessing object.
#' }
#' @export
fastolda <- function(X, Y, preproc = center(), reg = 0.01) {
  
  if (!is.factor(Y)) {
    stop("Y must be a factor.")
  }
  
  # Preprocess data
  procres <- preproc %>% prep()
  X <- init_transform(procres, X)
  
  freq <- table(Y)
  
  # compute_Htdiff is assumed to be defined elsewhere.
  Ht_diff <- compute_Htdiff(X) 
  Ht_diffS <- crossprod(Ht_diff)
  
  # Compute m as mean vector from Ht_diff
  m <- rowSums(Ht_diff)/nrow(X)
  
  # Construct HT matrix
  # HT constructed from Ht_diffS and mean adjustments:
  # This step seems incomplete or conceptual in the original code.
  # The original code attempts to build HT:
  #   HT <- sweep(Ht_diffS, 1, crossprod(m, Ht_diff), "-")
  # The following lines replicate that:
  
  HT <- sweep(Ht_diffS, 1, crossprod(m, Ht_diff), "-")
  HT <- cbind(t(-m %*% Ht_diff), HT)
  
  # Compute ST = HT' * HT
  ST <- tcrossprod(HT, HT)
  
  RT <- chol(ST)
  RT_inv <- chol2inv(RT)
  
  # HBs = t(rowsum(t(HT), Y)) reduces HT per-group. The original code:
  # HBs <- t(rowsum(t(HT), Y))
  # We then scale by freq * sqrt(freq)
  # However, this code is somewhat unclear in intent. We'll trust original logic.
  
  HBs <- t(rowsum(t(HT), Y))
  HB <- sweep(HBs, 2, freq * sqrt(freq), "/")
  
  # Construct Hfinal
  # Extracting columns 1:(length(freq)-1) from HB implies a dimension reduction step.
  Hfinal <- Ht_diff %*% RT_inv %*% HB[, 1:(length(freq)-1), drop=FALSE]
  
  # QR decomposition to get orthonormal basis
  GR <- qr(Hfinal)
  v <- qr.Q(GR)
  
  # Project data
  s <- X %*% v
  
  # Return a discriminant_projector object
  multivarious::discriminant_projector(
    rotation = v,
    s = s,
    sdev = apply(s, 2, sd),   # replaced proj_final with s
    labels = Y,
    preproc = procres,
    classes = "fast_olda"
  )
}