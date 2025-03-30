#' @title 
#' Stable and Orthogonal Local Discriminant Embedding using Trace Ratio Criterion (SOLDE-TR)
#'        with nanoflann-based k-NN for Fast Adjacency Construction
#'
#' @description
#' This is an extension of the SOLDE-TR algorithm described in:
#' \insertRef{yang2017stable}{}
#' 
#' The method extends the Stable Orthogonal Local Discriminant Embedding (SOLDE)
#' approach by recasting its objective function into a trace ratio optimization problem
#' and using an iterative trace-ratio (ITR) algorithm to jointly learn the orthogonal
#' projection vectors. Compared to the step-by-step SOLDE, SOLDE-TR converges faster,
#' yields a global solution, and usually provides improved performance in tasks such as
#' face recognition.
#'
#' \strong{What is new in this version?}  
#' We accelerate the adjacency graph construction (Steps 1--2 in Section 3 of the paper)
#' by using the \code{Rnanoflann} package, which provides a \code{nanoflann}-backed kd-tree
#' for extremely fast nearest neighbor (k-NN) queries. This can be a substantial improvement
#' over naive \eqn{O(N^2)} neighbor searches when \eqn{N} is large.
#'
#' \strong{Correspondence to the Paper Sections:}
#' \itemize{
#'   \item \strong{Equations (1), (2), (3):} Construction of adjacency (weight) matrices 
#'         \eqn{S}, \eqn{H}, \eqn{F} for similarity, diversity, and inter-class separability.
#'   \item \strong{Equations (4)--(6):} Objective functions in local discriminant embedding.
#'   \item \strong{Equation (7):} Combination into \eqn{L_d = \alpha L_s - (1-\alpha)L_v}.
#'   \item \strong{Equations (11)--(16):} The trace ratio formulation and iterative algorithm.
#' }
#'
#' @details
#' Main steps:
#' \enumerate{
#'   \item \strong{(Optional) PCA Projection:}
#'     We project \eqn{X} into a PCA subspace to ensure \eqn{S_d} is nonsingular.
#'
#'   \item \strong{Build Adjacency Graphs with \code{Rnanoflann}:}
#'     - \eqn{S} (similarity)  
#'       If two points \eqn{x_i} and \eqn{x_j} belong to the same class and either 
#'       \eqn{x_j} is among the \eqn{s} nearest neighbors of \eqn{x_i} \emph{or} 
#'       \eqn{x_i} is among the \eqn{s} neighbors of \eqn{x_j}, then 
#'       \eqn{S_{ij} = \exp(-\|x_i - x_j\|^2/\sigma)}.
#'     - \eqn{H} (diversity)  
#'       Same condition (same-class neighbors), but \eqn{H_{ij} = 1 - \exp(-\|x_i - x_j\|^2/\sigma)}.
#'     - \eqn{F} (interclass)  
#'       If \eqn{x_i} and \eqn{x_j} are from different classes \emph{and} they are in each
#'       other's \eqn{s} neighborhoods (union condition), 
#'       then \eqn{F_{ij} = \exp(-\|x_i - x_j\|^2/\sigma)}.
#' 
#'     We implement this neighbor search using \code{nanoflann}'s kd-tree via \code{nn()} from
#'     the \pkg{Rnanoflann} package for Euclidean distance.
#'
#'   \item \strong{Iterative Trace Ratio (ITR):}
#'     Solve
#'     \deqn{
#'       \max_{W^T W = I} \frac{\mathrm{tr}(W^T S_p W)}{\mathrm{tr}(W^T S_d W)},
#'     }
#'     with \eqn{S_p = X L_m X^T} and \eqn{S_d = X L_d X^T}, via the iterative approach:
#'     \eqn{\mathbf{M} = \mathbf{S_p} - \lambda \mathbf{S_d}}, eigen-decompose \(\mathbf{M}\),
#'     pick the top \eqn{m} eigenvectors.  Update \(\lambda\), repeat until convergence.
#'
#'   \item \strong{Final Embedding:}  
#'     \eqn{\mathbf{W} = \mathbf{W}_{\mathrm{PCA}} \times \mathbf{W}_{\mathrm{SOLDE-TR}}}.
#' }
#'
#' @param X A numeric matrix of size \eqn{N \times D} (row-wise samples).
#' @param y A numeric vector of length \eqn{N} (class labels).
#' @param s Number of neighbors for adjacency; default = 5.
#' @param alpha Numeric in \eqn{[0.5, 1]}. Balances \eqn{L_s} vs \eqn{L_v} in 
#'        \eqn{L_d = alpha * L_s - (1-alpha)*L_v}; default = 0.9.
#' @param sigma Positive scalar for the heat kernel \(\exp(-\|x_i - x_j\|^2 / \sigma)\); default = 1.0.
#' @param m Desired dimension of the subspace; default = 10.
#' @param tol Stopping threshold for change in trace ratio; default = 1e-6.
#' @param maxit Maximum number of ITR iterations; default = 100.
#' @param pca_preprocess Logical: whether to do PCA to ensure full rank; default = TRUE.
#' @return A list with:
#' \item{\code{W_pca}}{The PCA projection matrix (columns are the top PCA components).}
#' \item{\code{W_solde_tr}}{The \eqn{m} projection vectors from the trace ratio solution in the PCA subspace.}
#' \item{\code{W}}{The final \eqn{D x m} projection matrix = \(\mathbf{W}_{\mathrm{PCA}} \times \mathbf{W}_{\mathrm{SOLDE-TR}}\).}
#' \item{\code{eigvals_pca}}{Eigenvalues of the PCA step (if \code{pca_preprocess} = TRUE).}
#' \item{\code{trace_ratio_history}}{Vector of trace ratio values across ITR iterations.}
#'
#' @references 
#' \enumerate{
#'   \item Yang, X., Liu, G., Yu, Q., & Wang, R. (2017). 
#'         \emph{Stable and orthogonal local discriminant embedding using trace ratio criterion for dimensionality reduction.}
#'         Pattern Recognition, 71, 249--264.
#'   \item Wang, J., Zhu, X., & Gong, S. (2007). 
#'         \emph{ITR: Iterative trace ratio algorithm for feature extraction.} 
#'         IEEE Transactions on Knowledge and Data Engineering.
#'   \item He, X., Yan, S., Hu, Y., Niyogi, P., & Zhang, H. (2005). 
#'         \emph{Face recognition using Laplacianfaces.} 
#'         IEEE Trans. on Pattern Analysis and Machine Intelligence.
#'   \item \pkg{Rnanoflann} \url{https://github.com/ManosPapadakis95/Rnanoflann}
#' }
#'
#' @examples
#' \dontrun{
#'  # Simple test on a small dataset
#'  library(Rnanoflann)
#'  set.seed(1)
#'  N <- 100
#'  D <- 20
#'  X <- matrix(rnorm(N*D), nrow = N, ncol = D)
#'  y <- sample.int(3, size = N, replace = TRUE)
#'
#'  solde_res <- SOLDE_TR_fastNN(X, y, s=5, alpha=0.9, sigma=1.0, m=8, tol=1e-5, maxit=50)
#'  # Project new data X_new:
#'  # Ynew <- t(solde_res$W) %*% t(X_new)
#' }
#' @export
solde_tr <- function(X,
                            y,
                            s       = 5,
                            alpha   = 0.9,
                            sigma   = 1.0,
                            m       = 10,
                            tol     = 1e-6,
                            maxit   = 100,
                            pca_preprocess = TRUE) {
  ##############################################################################
  # 0. Checks and Packages
  ##############################################################################
  stopifnot(requireNamespace("Rnanoflann", quietly = TRUE))
  if (!is.matrix(X)) X <- as.matrix(X)
  if (length(y) != nrow(X)) {
    stop("Length of label vector y must match the number of rows in X.")
  }
  N <- nrow(X)
  D <- ncol(X)
  
  ##############################################################################
  # Helper function: Build adjacency matrix with Rnanoflann
  # (Equations (1), (2), (3))
  ##############################################################################
  build_adjacency <- function(Xpca, y, s, sigma, mode=c("similarity","diversity","interclass")) {
    mode <- match.arg(mode)
    nobs <- nrow(Xpca)
    W <- matrix(0, nrow = nobs, ncol = nobs)
    
    # Use Rnanoflann::nn() to find s+1 neighbors (the first neighbor is the point itself)
    # data=Xpca => from which we find neighbors
    # points=Xpca => for each row i, find neighbors in the data
    # We want Euclidean. We'll keep the distances for the heat kernel if needed.
    suppressWarnings(
      nn_out <- Rnanoflann::nn(
        data    = Xpca,
        points  = Xpca,
        k       = s+1,           # find s+1 neighbors
        method  = "euclidean",
        search  = "standard",
        eps     = 0.0,
        square  = FALSE,
        trans   = FALSE, # so that nn_out$indices has dimension (N x (s+1))
        sorted  = FALSE
      )
    )
    # nn_out$indices is N x (s+1)
    # nn_out$distances is N x (s+1)
    
    idx_mat  <- nn_out$indices
    dist_mat <- nn_out$distances
    
    # Fill adjacency
    for (i in seq_len(nobs)) {
      # The neighbors of i (excluding the first which should be i itself)
      neigh_idx  <- idx_mat[i, -1]
      neigh_dist <- dist_mat[i, -1]
      for (k_i in seq_along(neigh_idx)) {
        j  <- neigh_idx[k_i]
        dd <- neigh_dist[k_i]
        
        if (j < 1 || j > nobs) next  # safety check
        
        same_class <- (y[i] == y[j])
        if (mode == "similarity") {
          if (same_class) {
            # (1) S_ij = exp(-||xi - xj||^2 / sigma)
            kij <- exp(- dd^2 / sigma)
            W[i, j] <- max(W[i, j], kij)
            W[j, i] <- max(W[j, i], kij)  # union condition, symmetrical for Euclidean
          }
        } else if (mode == "diversity") {
          if (same_class) {
            # (2) H_ij = 1 - exp(-||xi - xj||^2 / sigma)
            kij <- exp(- dd^2 / sigma)
            val <- 1 - kij
            W[i, j] <- max(W[i, j], val)
            W[j, i] <- max(W[j, i], val)
          }
        } else if (mode == "interclass") {
          if (!same_class) {
            # (3) F_ij = exp(-||xi - xj||^2 / sigma)
            kij <- exp(- dd^2 / sigma)
            W[i, j] <- max(W[i, j], kij)
            W[j, i] <- max(W[j, i], kij)
          }
        }
      }
    }
    
    W
  }
  
  diag_colsum <- function(A) {
    # returns diag of column sums
    cc <- colSums(A)
    diag(cc)
  }
  
  ##############################################################################
  # 1. (Optional) PCA Projection (Section 3)
  ##############################################################################
  X_centered <- scale(X, center = TRUE, scale = FALSE)
  W_pca <- diag(1, D, D)  # identity if no PCA preprocessing
  eigvals_pca <- rep(1, D)
  
  if (pca_preprocess) {
    # Cov matrix (D x D)
    C <- crossprod(X_centered) / (N - 1)
    eC <- eigen(C, symmetric = TRUE)
    vals <- eC$values
    vecs <- eC$vectors
    
    # keep only positive
    pos_idx <- which(vals > 1e-12)
    vals <- vals[pos_idx]
    vecs <- vecs[, pos_idx, drop=FALSE]
    # sort desc
    o <- order(vals, decreasing=TRUE)
    vals <- vals[o]
    vecs <- vecs[, o, drop=FALSE]
    
    W_pca      <- vecs
    eigvals_pca <- vals
    
    # project X
    X_pca <- X_centered %*% W_pca  # N x PC
  } else {
    # If skipping PCA, we still center but do no dimension reduction
    X_pca <- X_centered
  }
  
  PC <- ncol(X_pca)
  if (PC < m) {
    stop("PCA subspace dimension < desired m. Reduce m or disable PCA.")
  }
  
  ##############################################################################
  # 2. Build adjacency graphs using nanoflann (Equations (1), (2), (3))
  ##############################################################################
  S <- build_adjacency(X_pca, y, s, sigma, mode="similarity")
  H <- build_adjacency(X_pca, y, s, sigma, mode="diversity")
  F <- build_adjacency(X_pca, y, s, sigma, mode="interclass")
  
  # Laplacians:
  # L_s = D^s - S
  Ds <- diag_colsum(S)
  L_s <- Ds - S
  
  # L_v = D^v - H
  Dv <- diag_colsum(H)
  L_v <- Dv - H
  
  # L_m = D^m - F
  Dm <- diag_colsum(F)
  L_m <- Dm - F
  
  # L_d = alpha * L_s - (1-alpha)* L_v   (Eq. (7))
  L_d <- alpha * L_s - (1 - alpha) * L_v
  
  # S_p = X_pca^T * L_m * X_pca
  # S_d = X_pca^T * L_d * X_pca
  S_p <- t(X_pca) %*% L_m %*% X_pca
  S_d_ <- t(X_pca) %*% L_d %*% X_pca
  # make them symmetric
  S_p   <- 0.5 * (S_p + t(S_p))
  S_d_  <- 0.5 * (S_d_ + t(S_d_))
  
  # small shift on diag to ensure pos-def
  diag(S_d_) <- diag(S_d_) + 1e-12
  
  ##############################################################################
  # 3. Iterative Trace Ratio (ITR) (Equations (13)-(16))
  ##############################################################################
  # random init for W in PCA subspace, size PC x m
  W0 <- matrix(rnorm(PC*m), nrow=PC, ncol=m)
  qrW0 <- qr(W0)
  Wk   <- qr.Q(qrW0)[, seq_len(m), drop=FALSE]
  
  trace_ratio_history <- numeric(maxit)
  
  for (iter in seq_len(maxit)) {
    # compute lambda_t
    tr_num <- sum(diag(t(Wk) %*% S_p  %*% Wk))
    tr_den <- sum(diag(t(Wk) %*% S_d_ %*% Wk))
    lambda_t <- tr_num / tr_den
    
    # M = S_p - lambda_t * S_d_
    M <- S_p - lambda_t * S_d_
    
    # eigen decomp
    eM <- eigen(M, symmetric=TRUE)
    vals_M <- eM$values
    vecs_M <- eM$vectors
    # top m
    idx_top <- order(vals_M, decreasing=TRUE)[1:m]
    Wnew <- vecs_M[, idx_top, drop=FALSE]
    # orthonormal
    qrWnew <- qr(Wnew)
    Wnew <- qr.Q(qrWnew)[, seq_len(m), drop=FALSE]
    
    trace_ratio_history[iter] <- lambda_t
    
    if (iter > 1) {
      if (abs(trace_ratio_history[iter] - trace_ratio_history[iter - 1]) < tol) {
        trace_ratio_history <- trace_ratio_history[seq_len(iter)]
        Wk <- Wnew
        break
      }
    }
    Wk <- Wnew
  }
  
  W_solde_tr <- Wk
  
  ##############################################################################
  # 4. Final Embedding
  #    W = W_pca * W_solde_tr
  ##############################################################################
  W_final <- W_pca %*% W_solde_tr  # (D x PC) * (PC x m) => D x m
  
  dp_obj <- multivarious::discriminant_projector(
    v       = W_final,         # (D x m)
    s       = X %*% W_final,   # training scores (N x m)
    sdev    = apply(X %*% W_final, 2, sd),
    preproc = your_preprocessing_object, # if you did any
    labels  = y,
    trace_ratio_history=trace_ratio_history,
    classes = "solde_tr"
  )
  return(dp_obj)
}