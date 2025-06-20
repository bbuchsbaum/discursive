#' Sparse or Ridge Low-Rank Regression (SLRR / LRRR)
#'
#' Implements a low-rank regression approach from:
#' \emph{"On The Equivalent of Low-Rank Regressions and Linear Discriminant Analysis Based Regressions"}
#' by Cai, Ding, and Huang (2013). This framework unifies:
#' \itemize{
#'   \item \strong{Low-Rank Ridge Regression (LRRR)}: when \code{penalty="ridge"},
#'         adds a Frobenius norm penalty \(\|\mathbf{A B}\|_F^2\).  
#'   \item \strong{Sparse Low-Rank Regression (SLRR)}: when \code{penalty="l21"},
#'         uses an \(\ell_{2,1}\) norm for row-sparsity. An iterative reweighting
#'         approach is performed to solve the non-smooth objective.
#' }
#'
#' In both cases, the model is equivalent to \emph{performing LDA-like dimensionality reduction}
#' (finding a subspace \(\mathbf{A}\) of rank \code{s}) and then doing a
#' \emph{regularized regression} (\(\mathbf{B}\)) in that subspace. The final
#' regression matrix is \(\mathbf{W} = \mathbf{A}\,\mathbf{B}\), which has rank
#' at most \code{s}.
#'
#' @param X A numeric matrix \(\mathrm{n \times d}\). Rows = samples, columns = features.
#' @param Y A factor or numeric vector of length \(\mathrm{n}\), representing class labels.
#'          If numeric, it will be converted to a factor.
#' @param s The rank (subspace dimension) of the low-rank coefficient matrix \(\mathbf{W}\).
#'          Must be \(\le\) the number of classes - 1, typically.
#' @param lambda A numeric penalty parameter (default 0.001).
#' @param penalty Either \code{"ridge"} for Low-Rank Ridge Regression (LRRR) or
#'                \code{"l21"} for Sparse Low-Rank Regression (SLRR).
#' @param max_iter Maximum number of iterations for the \code{"l21"} iterative algorithm.
#'                 Ignored if \code{penalty="ridge"} (no iteration needed).
#' @param tol Convergence tolerance for the iterative reweighting loop if \code{penalty="l21"}.
#' @param preproc A preprocessing function/object from \pkg{multivarious}, default \code{center()},
#'                to center (and possibly scale) \code{X} before regression.
#' @param verbose Logical, if \code{TRUE} prints iteration details for the \code{"l21"} case.
#' @param st_ridge Small ridge term added to \code{S_t} when solving the eigenproblem
#'        to avoid singularity (default 1e-6).
#'
#' @return A \code{\link[multivarious]{discriminant_projector}} object with subclass \code{"slrr"} that contains:
#'   \itemize{
#'     \item \code{v} : The \(\mathrm{d \times c}\) final regression matrix mapping original features to class-space
#'                      (\(\mathbf{W}\) in the paper). Here, \(\mathrm{c}\) = number of classes.
#'     \item \code{s} : The \(\mathrm{n \times c}\) score matrix (\(\mathbf{X}_\text{proc} \times \mathbf{W}\)).
#'     \item \code{sdev} : Standard deviations per column of \code{s}.
#'     \item \code{labels} : The factor labels \code{Y}.
#'     \item \code{preproc} : The preprocessing object used.
#'     \item \code{classes} : Will include \code{"slrr"} (and possibly \code{"ridge"} or \code{"l21"}).
#'     \item \code{A} : The learned subspace (optional debug). \(\mathrm{d \times s}\).
#'     \item \code{B} : The learned regression in subspace (optional debug). \(\mathrm{s \times c}\).
#'   }
#'
#' @details
#' \strong{1) Build Soft-Label Matrix:}  
#' We convert \code{Y} to a factor, then create an indicator matrix \(\mathbf{G}\)
#' with \code{nrow} = \code{n}, \code{ncol} = \code{c}, normalizing each column to sum to 1
#' (akin to the "normalized training indicator" in the paper).
#'
#' \strong{2) LDA-Like Subspace:}  
#' We compute total scatter \(\mathbf{S}_t\) and between-class scatter \(\mathbf{S}_b\),
#' then solve \(\mathbf{M} = \mathbf{S}_t^{-1} \mathbf{S}_b\) for its top \code{s} eigenvectors
#' \(\mathbf{A}\). This yields the rank-$s$ subspace.
#'
#' \strong{3) Regression in Subspace:}  
#' Let \(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{D}\) be the (regularized) covariance term
#' to invert, where:
#'   \itemize{
#'     \item If \code{penalty="ridge"}, \(\mathbf{D} = \mathbf{I}\).
#'     \item If \code{penalty="l21"}, we iterate a \emph{reweighted} diagonal \(\mathbf{D}\)
#'           to encourage row-sparsity (cf. the paper's Eq. (23-30)).
#'   }
#'
#' Then we solve \(\mathbf{B} = [\mathbf{A}^\top(\mathbf{X}\mathbf{X}^\top + \lambda\mathbf{D})\mathbf{A}]^{-1}
#'              [\mathbf{A}^\top\mathbf{X}\mathbf{G}].\)
#'
#' Finally, \(\mathbf{W} = \mathbf{A}\mathbf{B}\). We project the data \(\mathbf{X}_\text{proc}\)
#' to get scores \(\mathbf{X}_\text{proc}\mathbf{W}\).
#'
#' If \code{penalty="l21"}, we repeat the sub-steps for \(\mathbf{A},\mathbf{B}\) while updating
#' \(\mathbf{D}\) from the row norms of \(\mathbf{W}=\mathbf{A}\mathbf{B}\) in each iteration.
#' This leads to a row-sparse solution.
#'
#' @references
#' \itemize{
#'   \item Cai, X., Ding, C., & Huang, H. (2013). "On The Equivalent of Low-Rank Regressions and
#'   Linear Discriminant Analysis Based Regressions." \emph{KDD'13}.
#' }
#'
#' @export
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' Y <- iris[, 5]
#'
#' # Example 1: Low-Rank Ridge Regression (LRRR) with s=2, lambda=0.01
#' fit_lrrr <- slrr(X, Y, s=2, lambda=0.01, penalty="ridge")
#' print(fit_lrrr)
#'
#' # Example 2: Sparse Low-Rank Regression (SLRR) with l21 penalty
#' # and iterative approach, s=2, lambda=0.01
#' fit_slrr <- slrr(X, Y, s=2, lambda=0.01, penalty="l21", max_iter=20, tol=1e-5)
#' print(fit_slrr)
#' }
slrr <- function(X, 
                 Y, 
                 s, 
                 lambda = 0.001,
                 penalty = c("ridge", "l21"),
                 max_iter = 50,
                 tol = 1e-6,
                 preproc = center(),
                 verbose = FALSE,
                 st_ridge = 1e-6)
{
  penalty <- match.arg(penalty)
  
  # 1) Preprocess data
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)    # n x d
  n <- nrow(Xp)
  d <- ncol(Xp)
  
  # 2) Convert Y to factor, build soft-label matrix G
  Y <- as.factor(Y)
  G <- model.matrix(~ Y - 1)       # n x c (c = number of classes)
  # Normalize each column to sum=1
  G <- sweep(G, 2, colSums(G), FUN="/")
  cdim <- ncol(G)                  # number of classes

  if (cdim < 2) {
    stop("Y must contain at least two classes")
  }

  max_s <- min(cdim - 1, d)
  if (s < 1 || s > max_s) {
    stop(sprintf("s must be between 1 and %d", max_s))
  }

  # precompute cross-products used repeatedly
  Xt <- t(Xp)
  XtX <- Xt %*% Xp
  
  # 3) Compute total scatter (St) and between-class scatter (Sb)
  mu <- colMeans(Xp)
  St <- total_scatter(Xp, mu)      # d x d
  Sb <- between_class_scatter(Xp, Y) # d x d
  
  # 4) Solve generalized eigenproblem M = (St + st_ridge*I)^-1 Sb => top s eigenvectors => A
  St_reg <- St + st_ridge * diag(d)
  M <- tryCatch({
    solve(St_reg, Sb)
  }, error = function(e) {
    MASS::ginv(St_reg) %*% Sb
  })
  eigres <- eigen(M, symmetric=TRUE)
  A <- eigres$vectors[, 1:s, drop=FALSE]  # d x s
  
  # 5) Prepare to solve B in subspace, for the penalty approach
  #    We want to solve: B = [A^T (X X^T + lambda D) A]^-1 (A^T X G).
  #    If penalty="ridge", D = I (one-shot).
  #    If penalty="l21", we do iterative reweighting for D.
  
  # We'll define a small function to solve for (A, B, W) given D
  get_ABW <- function(A_in, D_in) {
    reg_mat <- XtX + lambda * D_in
    left_mat <- t(A_in) %*% reg_mat %*% A_in
    left_inv <- tryCatch({
      solve(left_mat)
    }, error = function(e) {
      MASS::ginv(left_mat)
    })
    B_in <- left_inv %*% (t(A_in) %*% Xt %*% G)  # (s x c)
    
    W_in <- A_in %*% B_in                        # (d x c)
    list(A=A_in, B=B_in, W=W_in)
  }
  
  # Initialize D => identity for both cases,
  # but in "l21" we iteratively update D from W's row norms
  Dmat <- diag(d)
  
  if (penalty == "ridge") {
    # Single pass => get B, W
    outABW <- get_ABW(A, Dmat)
    W <- outABW$W
    B <- outABW$B
  } else {
    # ========== "l21" iterative reweighting approach ==========
    # We'll define an iterative loop:
    # D_{i,i} = 1 / (2 * ||row_i(W)||_2) for i=1..d
    # Then solve for B => update A => actually, the paper's approach can re-solve A as well, 
    # but let's keep it consistent with eq. (30) if we want to re-estimate A each iteration. 
    # The paper's "Algorithm 3" suggests we update A and B in an inner loop.
    # 
    # For maximal faithfulness, we re-check the subspace each iteration:
    #   A step => eq. (30) => M = solve(St + lambda D, Sb)
    #   B step => eq. (28).
    
    # Possibly, we do:
    #   for t in 1:max_iter:
    #     # step 1: solve A => eq. (30) => top s eigenvectors of (St + lambda D)^-1 Sb
    #     # step 2: solve B => eq. (28)
    #     # step 3: update D => row norms of W
    # 
    # We'll track the objective (||Y - X^T W||^2 + lambda ||W||_{2,1}) for convergence.
    
    old_obj <- Inf
    W <- matrix(0, d, cdim)
    
    for (iter in seq_len(max_iter)) {
      # step 1) Solve for A => top s eigenvectors of (St + lambda D + st_ridge*I)^-1 Sb
      St_iter <- St + lambda * Dmat + st_ridge * diag(d)
      M_l21 <- tryCatch({
        solve(St_iter, Sb)
      }, error = function(e) {
        MASS::ginv(St_iter) %*% Sb
      })
      eig_l21 <- eigen(M_l21, symmetric=TRUE)
      A_new <- eig_l21$vectors[, 1:s, drop=FALSE]
      
      # step 2) Solve for B => eq. (28):
      # same as get_ABW but with the updated A
      reg_mat <- XtX + lambda * Dmat
      left_mat <- t(A_new) %*% reg_mat %*% A_new
      left_inv <- tryCatch({
        solve(left_mat)
      }, error = function(e) {
        MASS::ginv(left_mat)
      })
      B_new <- left_inv %*% (t(A_new) %*% Xt %*% G)
      
      W_new <- A_new %*% B_new
      
      # step 3) Update D => diag(1 / (2* rownorms(W_new)))
      # rownorm i = || row_i(W_new) ||_2
      row_norms <- sqrt(rowSums(W_new^2)) + 1e-12  # avoid zero
      D_diag <- 1 / (2 * row_norms)
      Dmat_new <- diag(D_diag, d)
      
      # compute objective => ||Y - X^T W||_F^2 + lambda * ||W||_{2,1}
      # X^T W => (d x n)(n x c)? Actually W is (d x c), so X^T W => (n x c)
      # but we have Xp is (n x d), so let's do s_mat = Xp %*% W => (n x c)
      s_mat_l21 <- Xp %*% W_new
      # residual
      R <- G - s_mat_l21
      fval1 <- sum(R^2)   # Fro norm^2
      # penalty = lambda * sum of row norms
      # row norms of W_new are row_norms
      fval2 <- lambda * sum(row_norms)
      new_obj <- fval1 + fval2
      
      if (verbose) {
        message(sprintf("Iter=%d, obj=%.6f (fval=%.6f + pen=%.6f)", 
                        iter, new_obj, fval1, fval2))
      }
      
      # check convergence
      if (abs(old_obj - new_obj) < tol * (1 + abs(old_obj))) {
        # converged
        A <- A_new
        W <- W_new
        B <- B_new
        Dmat <- Dmat_new
        break
      }
      
      # update
      A <- A_new
      W <- W_new
      B <- B_new
      Dmat <- Dmat_new
      old_obj <- new_obj
      
      if (iter == max_iter && verbose) {
        message("Max iterations reached in slrr l21 approach.")
      }
    } # end for
  }
  
  # 6) Build final scores => s = Xp %*% W => (n x c)
  s_mat <- Xp %*% W
  
  # 7) Return a discriminant_projector
  # Optionally store A,B if you'd like
  dp <- multivarious::discriminant_projector(
    v       = W,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = procres,
    labels  = Y,
    classes = c("slrr", penalty),
    A       = A,   # store subspace
    B       = B
  )
  
  return(dp)
}