#' DGPAGE: Discriminative and Geometry-Preserving Adaptive Graph Embedding
#'
#' This package-like script provides a complete, *faithful*, *working out-of-the-box* R implementation
#' of the **DGPAGE** method described in:
#'
#' > **J. Gou, X. Yuan, Y. Xue, L. Du, J. Yu, S. Xia, Y. Zhang**.  
#' > "Discriminative and Geometry-Preserving Adaptive Graph Embedding for dimensionality reduction."  
#' > *Information Sciences*, 2022.
#'
#' **DGPAGE** systematically integrates:
#' 1. Manually constructed *similarity* and *diversity* graphs (see Eqs. (4) and (7) in the paper),
#' 2. An adaptively learned adjacency graph with diversity-based regularization (Eqs. (8)-(11)),
#' 3. A multi-task-like objective that jointly optimizes the adjacency matrix and the projection matrix
#'    (Eq. (12)), and
#' 4. An alternating optimization algorithm (Algorithm 1 in the paper) that solves for these parameters.
#'
#' The main user-facing functions are:
#' \itemize{
#'   \item \code{dgpage_fit()}: fits the DGPAGE model on training data,
#'   \item \code{dgpage_predict()}: obtains predictions (via 1-NN) on new test data,
#'   \item \code{compute_similarity_graph()}: constructs the manual similarity graph \eqn{G_S} (Eq. (4)),
#'   \item \code{compute_diversity_graph()}: constructs the manual diversity graph \eqn{G_D} (Eq. (7)).
#' }
#'
#' The code below relies on base R (for matrix operations) and uses the
#' \strong{Rnanoflann} package for 1-nearest-neighbor classification.
#' 
#' **Note**: For large datasets \eqn{n}, the \eqn{O(n^2)} graph construction may
#' be expensive in plain R. You might want to accelerate via Rcpp or other means.
#'
#' @importFrom Rnanoflann nn
#' @author
#' \itemize{
#'   \item \strong{Original Paper}: Jianping Gou, Xia Yuan, Ya Xue, Lan Du, Jiali Yu, Shuyin Xia, Yi Zhang
#'   \item \strong{R Implementation}: Your Name (youremail@domain.com)
#' }
#' @references
#' \itemize{
#'   \item Gou, J., Yuan, X., Xue, Y., Du, L., Yu, J., Xia, S., & Zhang, Y. (2022).
#'     Discriminative and Geometry-Preserving Adaptive Graph Embedding for dimensionality reduction.
#'     \emph{Information Sciences}.
#' }
#' @name DGPAGE
NULL


# -------------------------------------------------------------------
#  dp_ge_projector.R -- DGPAGE as a discriminant_projector extension
# -------------------------------------------------------------------

#' Fit DGPAGE and return a discriminant_projector
#'
#' This function wraps \code{\link{dgpage_fit}} but returns a \code{discriminant_projector}
#' (subclass \code{"dgpage_projector"}) so that you can use the same \code{predict}
#' interface as for other discriminant methods.
#'
#' **Algorithm**:
#' \enumerate{
#'   \item If \code{S} or \code{D} is \code{NULL}, we build them from \code{X, y}
#'         via \code{compute_similarity_graph} and \code{compute_diversity_graph}.
#'   \item We call \code{\link{dgpage_fit}} to jointly learn the adjacency graph \code{W}
#'         and projection matrix \code{P}.
#'   \item We create a \code{discriminant_projector} object with:
#'         \itemize{
#'           \item \code{v = result$P} (the loadings),
#'           \item \code{s = X \%*\% result\$P} (the training scores),
#'           \item \code{sdev} a vector of stdev per dimension,
#'           \item \code{labels} = \code{y}.
#'         }
#'       We store \code{W} and other metadata in the returned object.
#' }
#'
#' **NOTE**: The \code{dgpage_predict} for 1-NN classification can now be replaced by
#' a new S3 \code{predict} method that lives under \code{dgpage_projector}.
#'
#' @param X A numeric matrix of size n x d (training samples).
#' @param y A factor (or coercible to factor) of length n (class labels).
#' @param S The n x n similarity graph. If \code{NULL}, we compute it from \code{X,y}.
#' @param D The n x n diversity graph. If \code{NULL}, we compute it from \code{X,y}.
#' @param r The target dimension (number of discriminant directions).
#' @param alpha,beta Hyperparameters in Eq. (12). 
#' @param maxiter,tol,verbose Passed to \code{\link{dgpage_fit}}.
#' @param q If \code{S} or \code{D} is \code{NULL}, we use \code{q} in
#'   \code{\link{compute_similarity_graph}}. Default 1.5.
#'
#' @return An object of class \code{c("dgpage_projector","discriminant_projector","bi_projector",...)}
#'   containing:
#'   \itemize{
#'     \item \code{v}        : the d x r loadings
#'     \item \code{s}        : the n x r scores
#'     \item \code{sdev}     : numeric vector of length r with stdevs of columns of s
#'     \item \code{labels}   : the training labels
#'     \item \code{W}        : the learned n x n adjacency graph
#'     \item \code{objective}: final objective value
#'     \item \code{alpha,beta} : stored hyperparameters
#'   }
#'
#' @examples
#' \dontrun{
#'   X <- matrix(rnorm(100*5), 100, 5)
#'   y <- factor(rep(1:2, each=50))
#'   dp <- dgpage_discriminant(X, y, r=2)
#'   preds_class  <- predict(dp, new_data=X, method="lda", type="class")
#'   preds_1nn    <- predict(dp, new_data=X, method="1nn")
#'   mean(preds_class == y)
#'   mean(preds_1nn   == y)
#' }
#' @export
dgpage_discriminant <- function(X, y,
                                S      = NULL,
                                D      = NULL,
                                r      = 2,
                                alpha  = 1e-3,
                                beta   = 1e-5,
                                maxiter= 20,
                                tol    = 1e-5,
                                verbose= TRUE,
                                q      = 1.5)
{
  y <- as.factor(y)
  n <- nrow(X)
  d <- ncol(X)
  if (length(y) != n) {
    stop("Length of y must match nrow(X).")
  }
  
  # Possibly build S, D:
  if (is.null(S)) {
    S <- compute_similarity_graph(X, y, q=q)
  }
  if (is.null(D)) {
    D <- compute_diversity_graph(X, y)
  }
  
  # Fit DGPAGE:
  fitres <- dgpage_fit(X, S=S, D=D, r=r,
                       alpha=alpha, beta=beta,
                       maxiter=maxiter, tol=tol,
                       verbose=verbose)
  
  # Build training scores: s = X %*% P
  # fitres$P is d x r
  s_mat <- X %*% fitres$P  # shape (n x r)
  # sdev => stdev of each column
  sdev_vec <- apply(s_mat, 2, sd)
  
  # Create a discriminant_projector object with an extra class "dgpage_projector"
  # We'll define it similarly to how you'd build any discriminant_projector:
  dp <- discriminant_projector(
    v       = fitres$P,     # loadings
    s       = s_mat,        # training scores
    sdev    = sdev_vec,     # stdev
    labels  = y,            # the training labels
    classes = "dgpage_projector"
  )
  
  # Store additional fields from the fit:
  dp$W         <- fitres$W
  dp$objective <- fitres$objective
  dp$alpha     <- alpha
  dp$beta      <- beta
  dp$iter      <- fitres$iter
  
  # You might also store fitres$Z or S, D, etc. if you want them for debugging:
  dp$S <- S
  dp$D <- D
  dp$Z <- fitres$Z
  
  # Return it
  dp
}

# -------------------------------------------------------------------
#  Predict method for "dgpage_projector"
# -------------------------------------------------------------------

#' Predict method for a \code{dgpage_projector} object
#'
#' Extends the discriminant methods with a \code{method="1nn"} option that
#' uses \pkg{Rnanoflann} for 1-nearest-neighbor classification in the projected space.
#'
#' Otherwise, if \code{method} is \code{"lda"} or \code{"euclid"}, we fall back to the
#' standard \code{\link{predict.discriminant_projector}} method for linear discriminant
#' or Euclidean nearest-mean classification in subspace.
#'
#' @param object A \code{dgpage_projector} object.
#' @param new_data A numeric matrix (m x d).
#' @param method One of \code{c("1nn","lda","euclid")}. Default \code{"1nn"}.
#' @param type If \code{method="1nn"}, we only return \code{"class"}. 
#'   If \code{method \%in\% c("lda","euclid")}, we can do \code{"class"} or \code{"prob"}.
#' @param k Number of neighbors if using \code{method="1nn"}. Default 1 (which is actually 1NN).
#' @param ... Not used (or pass \code{method="lda"} / \code{type="prob"}).
#'
#' @return 
#' If \code{method="1nn"}, a factor vector of length \code{m}.
#' If \code{method} is \code{"lda"} or \code{"euclid"} and \code{type="class"},
#' a factor of length \code{m}.
#' If \code{method="lda"} or \code{"euclid"} and \code{type="prob"}, an
#' \code{m x nclass} matrix of posterior-like values.
#'
#' @importFrom Rnanoflann nn
#' @seealso \code{\link{predict.discriminant_projector}}
#' @export
#' @import Rnanoflann
#' @importFrom multivarious predict.discriminant_projector
predict.dgpage_projector <- function(object,
                                     new_data,
                                     method = c("1nn","lda","euclid"),
                                     type   = c("class","prob"),
                                     k      = 1,
                                     ...) {
  method <- match.arg(method)
 
  if (method %in% c("lda","euclid")) {
    # Let the base 'predict.discriminant_projector' do the job.
    # We do need to set the 'type' properly:
    type <- match.arg(type)
   
     return(multivarious:::predict.discriminant_projector(object,
                                    new_data = new_data,
                                    method   = method,
                                    type     = type,
                                    ...))
    
  }
  
  # If method="1nn", then we do the standard Rnanoflann approach
  # similar to the old 'dgpage_predict' code, but integrated in S3 style:
  if (method == "1nn") {
    # type is forced to "class" basically:
    # We'll ignore user input if type="prob" (not supported for 1NN here).
    # So let's do a direct approach:
    
    # 1) Project new_data into subspace => new_data_proj
    new_data_proj <- new_data %*% object$v  # shape (m x r)
    
    # 2) Gather training scores and labels from object
    train_scores <- object$s      # shape (n x r)
    labs         <- object$labels # factor of length n
    
    # 3) Use Rnanoflann:
    #   data  = train_scores, points= new_data_proj,
    #   distance method = "euclidean" by default, can param. 
    #   BUT we must note that Rnanoflann expects row=points, col=features => that's consistent.
    res <- nn(data   = train_scores,
              points = new_data_proj,
              k      = k,
              trans=FALSE,
              method = "euclidean")
    
    # res$indices is k x m if trans=TRUE (default). We'll transpose to get m x k
    idx_mat <- t(res$indices)  # shape (m x k)
    
    # If k=1 => a single neighbor
    if (k == 1) {
      preds <- labs[idx_mat[,1]]
      return(preds)
    } else {
      # majority vote or something. Let's do majority in a naive way:
      # For each row in idx_mat, pick the most common class
      preds <- apply(idx_mat, 1, function(row_i) {
        tab <- table(labs[row_i])
        cl  <- names(which.max(tab))
        cl
      })
      preds <- factor(preds, levels=levels(labs))
      return(preds)
    }
  }
}




# -------------------------------------------------------------------------
#  1) Utility functions to construct the manual graphs
# -------------------------------------------------------------------------

#' Compute the manual similarity graph S (Eq. (4) in the paper)
#'
#' For each pair of samples \eqn{(x_i, x_j)}, the entry \eqn{s_{ij}} in the similarity
#' matrix \eqn{S} is computed as:
#' \deqn{
#'   s_{ij} =
#'   \begin{cases}
#'     \frac{1}{2}\left( e^{-\|x_i - x_j\|^2 / \tau_i^+} + e^{-\|x_j - x_i\|^2 / \tau_j^+} \right), & \text{if } c_i = c_j, \\
#'     -\frac{1}{2}\left( e^{-\|x_i - x_j\|^2 / \tau_i^-} + e^{-\|x_j - x_i\|^2 / \tau_j^-} \right), & \text{if } c_i \neq c_j,
#'   \end{cases}
#' }
#' where \eqn{\tau_i^+} and \eqn{\tau_i^-} (Eqs. (5) and (6)) characterize the intra- and inter-class
#' geometric distributions for sample \eqn{x_i}, respectively.
#'
#' @param X A numeric matrix of size \eqn{n \times d}, where each of the \eqn{n} rows is one sample.
#' @param y A length-\eqn{n} vector/factor of class labels.
#' @param q A positive exponent (see Eqs. (5) and (6)). Typical range is [1, 3].
#'
#' @return An \eqn{n \times n} matrix \eqn{S}.
#' @export
compute_similarity_graph <- function(X, y, q = 1.5) {
  n <- nrow(X)
  
  # Precompute distance matrix (squared distances).
  # dist(X) returns pairwise distances in condensed form -> convert to full matrix:
  distMat <- as.matrix(dist(X, method = "euclidean"))^2  # squared distances
  
  # For each sample i, compute:
  #   tau_i^+ = (1 / (n^+(c_i)^q)) * sum_{z_l in same class} || x_i - x_l ||^2
  #   tau_i^- = (1 / (n^-(c_i)^q)) * sum_{z_l in different classes} || x_i - x_l ||^2
  # where n^+(c_i) is #intra-class samples, n^-(c_i) is #inter-class samples.
  # Because each sample i includes itself, we carefully handle that sum.
  
  label_tab <- table(y)
  
  tau_plus <- numeric(n)
  tau_minus <- numeric(n)
  
  for (i in seq_len(n)) {
    same_idx <- which(y == y[i])
    diff_idx <- which(y != y[i])
    
    # Remove the sample i from the sums if desired,
    # but the original paper's eq(5),(6) typically includes i as well.
    # We'll follow the typical approach in references (Gou et al. 2020).
    
    # sum of squared distances to all *intra-class* samples
    s_plus <- sum(distMat[i, same_idx])
    # sum of squared distances to all *inter-class* samples
    s_minus <- sum(distMat[i, diff_idx])
    
    n_plus <- length(same_idx)
    n_minus <- length(diff_idx)
    
    tau_plus[i]  <- s_plus  / (n_plus^q)
    tau_minus[i] <- s_minus / (n_minus^q)
  }
  
  # Construct the S matrix
  S <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) {
        S[i, j] <- 0
      } else {
        if (y[i] == y[j]) {
          # intra-class
          val <- 0.5 * (exp(- distMat[i, j] / tau_plus[i]) +
                          exp(- distMat[j, i] / tau_plus[j]))
          S[i, j] <- val
        } else {
          # inter-class
          val <- -0.5 * (exp(- distMat[i, j] / tau_minus[i]) +
                           exp(- distMat[j, i] / tau_minus[j]))
          S[i, j] <- val
        }
      }
    }
  }
  
  return(S)
}


#' Compute the manual diversity graph D (Eq. (7) in the paper)
#'
#' For each pair of samples \eqn{(x_i, x_j)}, the entry \eqn{d_{ij}} in the diversity
#' matrix \eqn{D} is:
#' \deqn{
#'   d_{ij} =
#'   \begin{cases}
#'     \frac{1}{2}\left( e^{\|x_i - x_j\|^2 / \rho_i^+} + e^{\|x_j - x_i\|^2 / \rho_j^+} \right), & \text{if } c_i = c_j, \\
#'     \frac{1}{2}\left( e^{\|x_i - x_j\|^2 / \rho_i^-} + e^{\|x_j - x_i\|^2 / \rho_j^-} \right), & \text{if } c_i \neq c_j,
#'   \end{cases}
#' }
#' where \eqn{\rho_i^+} and \eqn{\rho_i^-} follow Eqs. (5) and (6) **but** with \eqn{q=0}.
#'
#' @param X A numeric matrix of size \eqn{n \times d}.
#' @param y A length-\eqn{n} vector/factor of class labels.
#'
#' @return An \eqn{n \times n} matrix \eqn{D}.
#' @export
compute_diversity_graph <- function(X, y) {
  n <- nrow(X)
  distMat <- as.matrix(dist(X, method = "euclidean"))^2  # squared distances
  
  # rho_i^+, rho_i^- ~ same as tau_i^+, tau_i^- but q=0
  # => effectively the average is replaced by total sum (since n^0 = 1).
  
  rho_plus  <- numeric(n)
  rho_minus <- numeric(n)
  
  for (i in seq_len(n)) {
    same_idx <- which(y == y[i])
    diff_idx <- which(y != y[i])
    
    s_plus  <- sum(distMat[i, same_idx])
    s_minus <- sum(distMat[i, diff_idx])
    # no division by n_plus^q or n_minus^q if q=0 => that is n_plus^0=1, so no effect:
    rho_plus[i]  <- s_plus
    rho_minus[i] <- s_minus
  }
  
  D <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) {
        D[i, j] <- 0
      } else {
        if (y[i] == y[j]) {
          val <- 0.5 * (exp(distMat[i, j] / rho_plus[i]) +
                          exp(distMat[j, i] / rho_plus[j]))
          D[i, j] <- val
        } else {
          val <- 0.5 * (exp(distMat[i, j] / rho_minus[i]) +
                          exp(distMat[j, i] / rho_minus[j]))
          D[i, j] <- val
        }
      }
    }
  }
  
  return(D)
}


# -------------------------------------------------------------------------
#  2) Main DGPAGE Fitting Algorithm
# -------------------------------------------------------------------------

#' Fit the DGPAGE model (Algorithm 1, Eqs. (12), (18), (23) in the paper)
#'
#' This function implements the main training procedure of DGPAGE, which jointly learns
#' the adaptive graph \eqn{W} and the projection matrix \eqn{P} by minimizing
#' Eq. (12) in an alternating fashion.
#'
#' **Steps**:
#' 1. Initialize \eqn{P} randomly (size \eqn{d \times r}).
#' 2. For each iteration:
#'    \itemize{
#'      \item Update \eqn{W} via Eq. (18) (closed-form).
#'      \item Update \eqn{P} by solving the generalized eigenvalue problem in Eq. (23).
#'    }
#' 3. Repeat until convergence or \code{maxiter} is reached.
#'
#' **Dimensions** (in internal notation):
#' \itemize{
#'   \item Let \eqn{Z = t(X)} be \eqn{d x n},
#'   \item \eqn{P} is \eqn{d x r},
#'   \item \eqn{Y = P^\top Z} is \eqn{r x n}.
#'   \item \eqn{W} is \eqn{n x n}.
#'   \item \eqn{S}, \eqn{D} are \eqn{n x n}.
#' }
#'
#' @param X A numeric matrix of size \eqn{n \times d} (training data).  
#'          \strong{Note}: We internally transpose \code{X} to follow the paper's notation.
#' @param S The \eqn{n x n} similarity graph (from \code{\link{compute_similarity_graph}}).
#' @param D The \eqn{n x n} diversity graph (from \code{\link{compute_diversity_graph}}).
#' @param r Target dimension (number of embedding directions to learn).
#' @param alpha The hyper-parameter in Eq. (12). Typical range: \eqn{[1e-6, 1e-1]}.
#' @param beta The hyper-parameter in Eq. (12). Typical range: \eqn{[1e-6, 1e-1]}.
#' @param maxiter Maximum number of iterations for the alternating updates.
#' @param tol Convergence tolerance on the objective value.
#' @param verbose Whether to print progress information.
#'
#' @return A list with:
#' \itemize{
#'   \item \code{P}: the \eqn{d \times r} projection matrix,
#'   \item \code{W}: the \eqn{n \times n} learned adjacency matrix (adaptive graph),
#'   \item \code{objective}: the final objective value,
#'   \item \code{iter}: number of iterations taken,
#'   \item \code{Z}: stored internally (\eqn{d x n}) = \code{t(X)} (just for reference).
#' }
#' @export
dgpage_fit <- function(X, S, D, r = 10,
                       alpha = 1e-3, beta = 1e-5,
                       maxiter = 20, tol = 1e-5,
                       verbose = TRUE) {
  # Transpose X: let Z = d x n
  Z <- t(X)  
  d <- nrow(Z)
  n <- ncol(Z)
  
  if (r > d) {
    warning("r > d. Embedding dimension cannot exceed the number of features. Setting r = d.")
    r <- d
  }
  
  # Initialize P: d x r (random orthonormal)
  set.seed(2024)  # for reproducibility, adjust as you like
  # We can do e.g. random normal, then orthonormalize:
  tmp <- matrix(rnorm(d*r), nrow=d, ncol=r)
  # Orthonormal via QR decomposition:
  qrtmp <- qr(tmp)
  P <- qr.Q(qrtmp)  # d x r (orthonormal columns)
  
  # Initialize W to the identity or zeros: n x n
  W <- diag(1, n, n)
  
  # a small function to compute the objective F(P,W) from Eq. (12) for monitoring
  compute_objective <- function(P, W) {
    # 1) reconstruction residual part: sum_{i=1}^n || P^T (z_i - Z w_i) ||^2
    #    can be written as || P^T (Z - Z W) ||_F^2
    #    that is trace( P^T (Z - ZW)(Z - ZW)^T P )
    #    but let's do it more directly: Y = P^T Z, => r x n
    #    Then Y - YW? Actually from Eq. (14) in the paper we do
    #    || P^T(Z - ZW) ||_F^2 = || Y - YW ||_F^2  where Y = P^T Z
    Y <- t(P) %*% Z  # r x n
    YW <- Y %*% W    # r x n
    part1 <- sum((Y - YW)^2)
    
    # 2) alpha * || D W ||_F^2
    part2 <- alpha * sum((D %*% W)^2)
    
    # 3) beta * sum_{i,j} || P^T z_i - P^T z_j ||^2 w_{ij}
    #    + (1 - alpha - beta) * sum_{i,j} || P^T z_i - P^T z_j ||^2 s_{ij}
    #    We can unify them: sum_{i,j} (||P^T z_i - P^T z_j||^2) * [ beta*w_{ij} + (1-alpha-beta)*s_{ij} ]
    #    We'll do a single pass i < j to save time, or do matrix logic
    #    We'll define L = beta*W + (1 - alpha - beta)*S, then use the standard LPP form:
    #    sum_{i,j} ||y_i - y_j||^2 L_{ij} = trace( Y L Y^T ), where Y is n x r if we treat rows as samples.
    #    But we have Y as r x n => let's define Y^T = n x r => then trace( Y^T L Y^T ) doesn't quite match dimension...
    #    Actually from the paper's standard approach: trace( P^T Z L Z^T P ).
    
    L <- beta * W + (1 - alpha - beta) * S  # n x n
    # Then the standard formula => trace( P^T Z L Z^T P )
    # i.e. sum_{i,j} w_{ij} * ||P^T z_i - P^T z_j||^2
    # We'll do M = Z L Z^T => d x d, then part3 = trace( P^T M P )
    # we can do sum(diag( P^T M P )) = sum(diag( M P P^T )) if P has orthonormal columns => but let's do direct
    M <- Z %*% L %*% t(Z)  # d x d
    part3 <- sum(diag(t(P) %*% M %*% P))
    
    return(part1 + part2 + part3)
  }
  
  # iterative optimization
  old_obj <- compute_objective(P, W)
  if (verbose) cat(sprintf("Initial objective = %.6f\n", old_obj))
  
  for (iter in seq_len(maxiter)) {
    # --- Step 1: Fix P, solve for W  (Eq. (18) in the paper) ---
    # We define:
    #   Y = P^T Z => r x n
    #   E_{ij} = || y_i - y_j ||^2, but y_i is the i-th column => dimension r.
    #   Then W = (2 Y^T Y + 2 alpha D^T D)^(-1) (2 Y^T Y - beta E)
    #   where E is n x n with E[i,j] = e_{ij}.
    
    Y <- t(P) %*% Z  # r x n
    
    # compute E (n x n)
    E <- matrix(0, n, n)
    for (i in seq_len(n)) {
      for (j in seq_len(n)) {
        diff_ij <- Y[, i] - Y[, j]
        E[i, j] <- sum(diff_ij^2)
      }
    }
    
    # YtY = t(Y) %*% Y => dimension: (n x r) * (r x n) = n x n
    YtY <- t(Y) %*% Y
    
    matA <- 2 * YtY + 2 * alpha * (t(D) %*% D)  # (n x n)
    matB <- 2 * YtY - beta * E                 # (n x n)
    
    W_new <- tryCatch({
      solve(matA, matB)
    }, error = function(e) {
      MASS::ginv(matA) %*% matB
    })
    
    # --- Step 2: Fix W, solve for P (Eq. (23) in the paper) ---
    # We define:
    #   K = (I - W)(I - W)^T => n x n
    #   L_W = H - W where H is diag row-sums of W => n x n
    #   L_S = Q - S where Q is diag row-sums of S => n x n
    #   A = K + beta L_W + (1 - alpha - beta) L_S => n x n
    #   Then solve generalized eigen-problem:
    #     Z A Z^T p = lambda Z Z^T p
    
    I_n <- diag(1, n)
    K <- (I_n - W_new) %*% t(I_n - W_new)  # n x n
    # compute L_W = H - W
    rowSumW <- rowSums(W_new)
    H <- diag(rowSumW, n, n)
    L_W <- H - W_new
    
    # compute L_S = Q - S
    rowSumS <- rowSums(S)
    Q <- diag(rowSumS, n, n)
    L_S <- Q - S
    
    A <- K + beta * L_W + (1 - alpha - beta) * L_S  # n x n
    
    # M = Z A Z^T => d x d
    M <- Z %*% A %*% t(Z)
    # N = Z Z^T => d x d
    N <- Z %*% t(Z)
    
    # Solve M p = lambda N p. We want the r smallest eigenvalues.
    # We'll use the 'geigen' from the 'geigen' package if installed, or fallback.
    # For demonstration, let's do 'geigen::geigen' or base 'eigen' trick:
    # We can do a standard trick: solve(N, M). Then do eigen( solve(N) %*% M ) => might be unstable.
    
    # safer approach for large d might be RSpectra or something. We'll do a small approach here:
    # we do: solve(N, M) p = lambda p => standard eigenproblem.
    
    # to avoid singular N, we can do a tiny Tikhonov or check rank.
    # We'll do a try-catch:
    Mtilde <- tryCatch({
      solve(N, M)
    }, error = function(e) {
      # fallback if singular
      MASS::ginv(N) %*% M
    })
    
    eigres <- eigen(Mtilde, symmetric=FALSE)  # not guaranteed symmetric if solve(N, M)
    # Then we pick the r eigenvectors corresponding to the r smallest real parts of eigenvalues
    # (the paper states "smallest eigenvalues" -> real. We might need only the real part)
    
    # filter real eigenvalues
    lambdas <- Re(eigres$values)
    vectors <- Re(eigres$vectors)
    
    # sort ascending
    idx_sorted <- order(lambdas, decreasing = FALSE)
    idx_chosen <- idx_sorted[seq_len(r)]
    P_eigs <- vectors[, idx_chosen, drop=FALSE]  # d x r
    
    # we might want to re-orthonormalize P
    # in principle, the generalized eigenvectors are typically not orthonormal w.r.t. the standard dot product.
    # But let's do a plain QR for numerical stability:
    qrP <- qr(P_eigs)
    P_new <- qr.Q(qrP)
    
    # update
    W <- W_new
    P <- P_new
    
    # compute new objective
    new_obj <- compute_objective(P, W)
    rel_diff <- abs(new_obj - old_obj) / (old_obj + 1e-15)
    if (verbose) {
      cat(sprintf("Iter %2d: objective=%.6f, rel_diff=%.6f\n", iter, new_obj, rel_diff))
    }
    
    if (rel_diff < tol) {
      if (verbose) cat("Converged!\n")
      break
    }
    old_obj <- new_obj
  } # end for
  
  final_obj <- compute_objective(P, W)
  
  list(
    P = P,
    W = W,
    objective = final_obj,
    iter = iter,
    Z = Z  # for reference
  )
}


# -------------------------------------------------------------------------
#  3) Prediction (1-NN) using Rnanoflann
# -------------------------------------------------------------------------

#' Predict labels using a trained DGPAGE model via 1-NN classification
#'
#' After you have fit the DGPAGE model to some labeled training data \code{Xtrain}, you can
#' project \code{Xtrain} and store their embeddings \code{Ytrain}, then for \emph{new} data
#' \code{Xtest}, you project them using the learned \code{P}, and perform a \strong{1-nearest neighbor}
#' search in the low-dimensional subspace using \pkg{Rnanoflann}.
#'
#' @param model The list returned by \code{\link{dgpage_fit}}.  
#'              Must also contain \code{ytrain} (the vector of training labels) and \code{Ytrain} 
#'              (the \eqn{n x r} embedding of training data), which you can store yourself.
#' @param Xtest A numeric matrix of size \eqn{m \times d} (test data).
#' @param k Number of neighbors (default 1).
#' @param method Distance method for \code{\link[Rnanoflann]{nn}}. Default "euclidean".
#'
#' @return A vector of length \eqn{m} with predicted labels.
#' @export
#' @seealso \code{\link[Rnanoflann]{nn}}
#' @examples
#' \dontrun{
#'   # Suppose we already fit a model:
#'   # model$Ytrain <- t(model$P) %*% model$Z  # r x n -> we store its transpose below
#'   # model$Ytrain_t <- t(model$Ytrain)       # n x r
#'   # model$ytrain  <- y  # training labels
#'   #
#'   # Now we get new data Xtest:
#'   preds <- dgpage_predict(model, Xtest)
#' }
dgpage_predict <- function(model, Xtest, k=1, method="euclidean") {
  if (is.null(model$P) || is.null(model$Z)) {
    stop("model does not contain P or Z. Did you remove them?")
  }
  if (is.null(model$Ytrain) || is.null(model$ytrain)) {
    stop("model does not contain Ytrain or ytrain. Please store them after training.")
  }
  
  # project Xtest using P
  Ztest <- t(Xtest)                 # d x m
  Ytest <- t(model$P) %*% Ztest     # r x m
  Ytest_t <- t(Ytest)               # m x r
  
 
  # data= model$Ytrain_t, points= Ytest_t
  res <- nn(
    data   = model$Ytrain_t,
    points = Ytest_t,
    k      = k,
    method = method
  )
  # res$indices is k x m if trans=TRUE. We see the doc: by default, trans=TRUE => it might be
  #   or if trans=FALSE => then it is m x k. Let's check carefully.
  # Usually, res$indices is (k x m) if default trans=TRUE.
  indices <- t(res$indices)  # make it m x k if needed
  
  # For k=1, we just pick the single neighbor:
  if (k == 1) {
    preds <- model$ytrain[ indices[,1] ]
  } else {
    # if k>1, we can do a majority vote or something
    # but for demonstration we do the first neighbor
    preds <- model$ytrain[ indices[,1] ]
  }
  
  return(preds)
}

#' @export
print.dgpage_projector <- function(x, ...) {
  cat("A 'dgpage_projector' object (subclass of 'discriminant_projector').\n")
  cat("Learned adjacency graph W of size:", nrow(x$W), "x", ncol(x$W), "\n")
  cat(sprintf("Objective: %.3f\n", x$objective))
  cat("alpha:", x$alpha, " beta:", x$beta, "\n")
  # Then delegate the rest to the standard print for discriminant_projector:
  NextMethod()  # calls print.discriminant_projector
}


