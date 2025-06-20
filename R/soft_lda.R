#' Compute Weighted Group Means
#'
#' Given a data matrix \code{X} (size \code{n x d}) and a class-weight matrix \code{F} (size \code{c x n}),
#' this function computes the weighted means of \code{X} for each class.
#' Unlike hard (0/1) assignments, each sample contributes partially to each class's mean 
#' according to its weight.
#'
#' @param X A numeric matrix of shape \code{n x d}, where \code{n} is the number of samples (rows)
#'          and \code{d} is the number of features (columns). If not already a
#'          matrix, it will be coerced with \code{as.matrix()}.
#' @param F A numeric matrix of shape \code{c x n}, where \code{c} is the number of classes
#'          and \code{n} is the number of samples. \code{F[i, j]} is the weight of sample \code{j}
#'          for class \code{i}. Each row must have a positive sum of weights; the
#'          function stops with an error otherwise. Non-matrix inputs are
#'          coerced with \code{as.matrix()}.
#'
#' @return A numeric matrix of shape \code{c x d}, where each row is the weighted mean of \code{X}
#'         for one class.
#'
#' @examples
#' # Suppose we have 5 samples (rows), 2 features (cols),
#' # and 3 classes. F has shape (3 x 5).
#' X <- matrix(1:10, nrow=5, ncol=2)
#' F <- matrix(runif(3*5, min=0, max=1), nrow=3)
#' # Ensure each row of F sums to something > 0
#' res_means <- weighted_group_means(X, F)
#' 
#' @export
weighted_group_means <- function(X, F) {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(F)) F <- as.matrix(F)

  row_sums_F <- rowSums(F)
  if (any(row_sums_F <= 0)) {
    stop("Each row of F must have a positive sum of weights (>=1 sample).")
  }

  # Vectorized computation: (c x n) %*% (n x d) => (c x d)
  ret <- F %*% X
  ret <- sweep(ret, 1, row_sums_F, "/")

  rownames(ret) <- rownames(F)
  colnames(ret) <- colnames(X)
  ret
}


#' Soft-Label Linear Discriminant Analysis (SL-LDA)
#'
#' This function implements a \emph{soft-label} variant of Linear Discriminant Analysis (LDA),
#' following the approach described in:
#' 
#' Zhao, M., Zhang, Z., Chow, T.W.S., & Li, B. (2014). 
#' "A general soft label based Linear Discriminant Analysis for semi-supervised dimensionality reduction."
#' \emph{Neurocomputing, 135}, 250-264.
#'
#' Instead of hard (0/1) labels, each sample can have fractional memberships (soft labels) across
#' multiple classes. These memberships are encoded in a matrix \code{C}, typically obtained via
#' a label-propagation or fuzzy labeling step. SL-LDA uses these soft memberships to form
#' generalized scatter matrices \(\widetilde{S}_w\) and \(\widetilde{S}_b\), then solves
#' an LDA-like dimension-reduction problem in a PCA subspace via a \emph{two-step} approach:
#'
#' \enumerate{
#'   \item \strong{Preprocessing}:
#'         Apply a \code{preproc} function (e.g. \code{center()}) to the data \code{X}.
#'   \item \strong{PCA}:
#'         Project the data onto the top \code{dp} principal components (to handle rank deficiency).
#'   \item \strong{Compute Soft-Label Scatter} in the PCA space:
#'         \itemize{
#'           \item Let \(\mathbf{F} = \mathbf{C}^\top\) be size \(\mathrm{c \times n}\).
#'           \item Let \(\mathbf{E} = \mathrm{diag}(\text{rowSums}(\mathbf{C}))\) (size \(\mathrm{n \times n}\)).
#'           \item Let \(\mathbf{G} = \mathrm{diag}(\text{colSums}(\mathbf{C}))\) (size \(\mathrm{c \times c}\)).
#'           \item Form \(\widetilde{S}_w = X_p^\top ( E - F^\top G^{-1} F ) X_p + \alpha I\) (within-class),
#'                 and \(\widetilde{S}_b = X_p^\top \bigl(F^\top G^{-1}F - \tfrac{E e e^\top E}{e\,E\,e^\top}\bigr) X_p\) (between-class).
#'         }
#'   \item \strong{Within-class projection} (\code{di}):
#'         Partially diagonalize \(\widetilde{S}_w\). 
#'         In code, we extract \code{di} eigenvectors. 
#'         (\emph{Note}: Some references keep the \emph{largest} eigenvalues, others the \emph{smallest}.)
#'   \item \strong{Between-class projection} (\code{dl}):
#'         Project the (soft) class means into the \code{di}-dim subspace, then run a small PCA
#'         for dimension \code{dl}.
#'   \item \strong{Combine}:
#'         Multiply \(\mathrm{(d \times dp)} \cdot (\mathrm{dp \times di}) \cdot (\mathrm{di \times dl})\)
#'         to get the final \(\mathrm{(d \times dl)}\) projection matrix.
#' }
#'
#' @param X A numeric matrix \(\mathrm{n \times d}\), rows = samples, columns = features.
#' @param C A numeric matrix \(\mathrm{n \times c}\) of soft memberships. 
#'          \code{C[i, j]} = weight of sample \code{i} for class \code{j}. 
#'          Must be \(\ge 0\); row sums can be any positive value 
#'          (if \(\sum_j C[i,j] = 1\), each row is a probability distribution).
#' @param preproc A \code{pre_processor} from \pkg{multivarious}, e.g. \code{center()} or \code{pass()}.
#'                Defaults to \code{pass()} (no centering).
#' @param dp Integer. Number of principal components to keep in the first PCA step. 
#'           Defaults to \code{min(dim(X))}.
#' @param di Integer. Dimension of the \emph{within-class} subspace. Default \code{dp - 1}.
#' @param dl Integer. Dimension of the final subspace for \emph{between-class} separation. 
#'           Default \code{ncol(C) - 1}.
#' @param alpha A numeric ridge parameter (\(\ge 0\)). If \code{alpha > 0}, we add \(\alpha I\) 
#'              to \(\widetilde{S}_w\) to ensure invertibility. Default \code{0}.
#'
#' @return A \code{\link[multivarious]{discriminant_projector}} object with subclass 
#'         \code{"soft_lda"} containing:
#' \itemize{
#'   \item \code{v} ~ The \(\mathrm{(d \times dl)}\) final projection matrix.
#'   \item \code{s} ~ The \(\mathrm{(n \times dl)}\) projected scores of the training set.
#'   \item \code{sdev} ~ The std dev of each dimension in \code{s}.
#'   \item \code{labels} ~ Currently set to \code{colnames(C)} (or \code{NULL}).
#'   \item \code{preproc} ~ The preprocessing object used.
#'   \item \code{classes} ~ A string \code{"soft_lda"}.
#' }
#'
#' @details
#' In typical references, one might pick the \emph{largest} eigenvalues of \(\widetilde{S}_w\) 
#' for stable inversion, but certain versions (like Null-LDA) use the \emph{smallest} eigenvalues.
#' Adjust the code in \code{RSpectra::eigs_sym()} accordingly if you prefer a different variant.
#'
#' If you want to confirm \(\widetilde{S}_t = \widetilde{S}_w + \widetilde{S}_b\) numerically, 
#' you can define a helper function for \(\widetilde{S}_t\) and compare it to \(\widetilde{S}_w + \widetilde{S}_b\).
#'
#' @references 
#' Zhao, M., Zhang, Z., Chow, T.W.S., & Li, B. (2014). 
#' "A general soft label based Linear Discriminant Analysis for semi-supervised dimensionality reduction."
#' \emph{Neurocomputing, 135}, 250-264.
#' 
#' @export
soft_lda <- function(X,
                     C,
                     preproc = pass(),
                     dp = min(dim(X)),
                     di = dp - 1,
                     dl = ncol(C) - 1,
                     alpha = 0.0)
{
  ## 1) Basic checks
  if (!is.matrix(X)) {
    stop("X must be a matrix.")
  }
  if (!is.matrix(C)) {
    stop("C must be a matrix of soft membership weights.")
  }
  if (nrow(C) != nrow(X)) {
    stop("C must have the same number of rows as X. (Each row of C => one sample.)")
  }
  if (any(C < 0)) {
    stop("All entries of 'C' must be non-negative (soft memberships).")
  }
  
  n <- nrow(X)
  d <- ncol(X)
  c <- ncol(C)  # number of classes
  
  ## 2) Preprocessing => typically center or pass
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)  # shape (n x d)
  
  ## 3) PCA => reduce to dp 
  pca_red <- multivarious::pca(Xp, ncomp = dp)
  proj_dp <- pca_red$v        # (d x dp) loadings
  Xpca    <- scores(pca_red)  # (n x dp)
  
  ## 4) Build the weighting matrices 
  F <- t(C)    # shape (c x n)
  E <- Matrix::Diagonal(x = rowSums(C))  # (n x n)
  G <- diag(colSums(C))                  # (c x c)
  
  e_vec <- rep(1, n)
  denom <- as.numeric(t(e_vec) %*% diag(E) %*% e_vec)  # e E e^T => total mass
  
  ## 5) Weighted scatter in PCA space
  #    within-class => Sw = Xpca^T [ E - F^T G^-1 F ] Xpca + alpha I
  #    between-class => Sb = Xpca^T [F^T G^-1 F - (E e e^T E)/denom ] Xpca

  Ft   <- t(F)
  invG <- diag(1 / diag(G))
  M_sw <- E - Ft %*% invG %*% F
  M_sb <- Ft %*% invG %*% F - (E %*% tcrossprod(e_vec) %*% E) / denom

  Sw_raw <- crossprod(Xpca, M_sw %*% Xpca)
  Sb_raw <- crossprod(Xpca, M_sb %*% Xpca)
  
  # add alpha => ridge for invertibility
  Sw <- Sw_raw + alpha * diag(dp)
  
  ## 6) Two-step LDA approach
  # Step A: partial diagonalization of Sw => dimension di
  if (di > dp) {
    warning("di > dp is invalid; setting di = dp.")
    di <- dp
  }
  if (di < 1) stop("di must be >= 1.")
  
  # Typically for stable "whitening", many references pick the largest eigenvalues => which="LM"
  # but some do "SM". Adjust as needed.
  E_i <- RSpectra::eigs_sym(Sw, k = di, which = "LM")  # <--- largest or smallest
  proj_di <- E_i$vectors %*% diag(1 / sqrt(E_i$values))  # whiten
  
  # Weighted group means in PCA space
  group_means_pca <- weighted_group_means(Xpca, F)  # c x dp
  gm_proj <- group_means_pca %*% proj_di            # c x di
  
  # Step B: small PCA for dl
  if (dl > di) {
    warning("dl > di is invalid; setting dl = di.")
    dl <- di
  }
  if (dl < 1) stop("dl must be >= 1.")
  
  E_l <- multivarious::pca(gm_proj, ncomp = dl)
  proj_dl <- E_l$v  # (di x dl)
  
  proj_final <- proj_dp %*% proj_di %*% proj_dl  # (d x dl)
  
  # build final training scores => (n x dl)
  s_mat <- Xpca %*% proj_di %*% proj_dl
  
  # return
  dp_obj <- multivarious::discriminant_projector(
    v       = proj_final,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = procres,
    labels  = colnames(C),
    classes = "soft_lda"
  )
  return(dp_obj)
}

