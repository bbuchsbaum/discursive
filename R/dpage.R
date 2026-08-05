#' DGPAGE: Discriminative and Geometry-Preserving Adaptive Graph Embedding
#'
#' An R implementation of the **DGPAGE** method. DGPAGE systematically integrates:
#' 1. Manually constructed *similarity* and *diversity* graphs (see Eqs. (4) and (7) in the paper),
#' 2. An adaptively learned adjacency graph with diversity-based regularization (Eqs. (8)-(11)),
#' 3. A multi-task-like objective that jointly optimizes the adjacency matrix and the projection matrix
#'    (Eq. (12)), and
#' 4. An alternating optimization algorithm (Algorithm 1 in the paper) that solves for these parameters.
#'
#' The main user-facing functions are:
#' \itemize{
#'   \item \code{dgpage_fit()}: fits the DGPAGE model on training data,
#'   \item \code{dgpage_predict()}: obtains predictions (via k-NN) on new test data,
#'   \item \code{compute_similarity_graph()}: constructs the manual similarity graph \eqn{G_S} (Eq. (4)),
#'   \item \code{compute_diversity_graph()}: constructs the manual diversity graph \eqn{G_D} (Eq. (7)).
#' }
#'
#' The code relies on base R for matrix operations and uses the
#' \strong{Rnanoflann} package for nearest-neighbor classification.
#'
#' **Note**: Graph construction is \eqn{O(n^2)} in time and memory, and each
#' iteration of \code{dgpage_fit} solves an \eqn{n \times n} linear system
#' (\eqn{O(n^3)}). Projection eigenproblems are solved in the numerical row
#' space of \code{X}, avoiding redundant feature-null directions when \eqn{d > n}.
#'
#' @importFrom Rnanoflann nn
#' @references
#' Gou, J., Yuan, X., Xue, Y., Du, L., Yu, J., Xia, S., & Zhang, Y. (2023).
#' Discriminative and Geometry-Preserving Adaptive Graph Embedding for
#' dimensionality reduction. \emph{Neural Networks}, 157, 364-376.
#' @name DGPAGE
NULL


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

.dgpage_is_finite_scalar <- function(x) {
  is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x)
}

.dgpage_is_positive_integer <- function(x) {
  .dgpage_is_finite_scalar(x) && x >= 1 && x == round(x)
}

# Validate labels for graph construction: at least two classes, and every
# class needs >= 2 members (a singleton class makes tau_plus/rho_plus zero,
# silently zeroing that sample's intra-class similarities).
.validate_dgpage_labels <- function(y, n) {
  assertthat::assert_that(length(y) == n, msg = "Length of y must match nrow(X).")
  assertthat::assert_that(!anyNA(y), msg = "y must not contain missing values.")
  y <- droplevels(as.factor(y))
  counts <- table(y)
  assertthat::assert_that(length(counts) >= 2, msg = "y must contain at least two classes.")
  assertthat::assert_that(min(counts) >= 2, msg = "Every class must contain at least two samples.")
  y
}

# Generalized (two-sided) Laplacian-like matrix for a possibly asymmetric W:
#   G = diag(rowSums(W) + colSums(W)) - W - t(W)
# satisfies sum_ij w_ij ||y_i - y_j||^2 = tr(Y G Y^T) for Y with samples as columns.
# For symmetric W this equals twice the usual Laplacian.
.pairwise_laplacian <- function(W) {
  n <- nrow(W)
  G <- -(W + t(W))
  diag(G) <- diag(G) + rowSums(W) + colSums(W)
  G
}

# Majority vote over a m x k matrix of training-set neighbor indices.
# Ties are broken by factor level order.
.knn_vote <- function(labs, idx_mat) {
  labs <- as.factor(labs)
  idx_mat <- as.matrix(idx_mat)
  if (ncol(idx_mat) == 1L) {
    return(labs[idx_mat[, 1L]])
  }
  votes <- apply(idx_mat, 1L, function(row_i) {
    tab <- table(labs[row_i])
    names(tab)[which.max(tab)]
  })
  factor(votes, levels = levels(labs))
}


.validate_knn_k <- function(k, n) {
  assertthat::assert_that(
    .dgpage_is_positive_integer(k) && k <= n,
    msg = paste0("k must be a positive integer no larger than the number of training samples (", n, ").")
  )
  as.integer(k)
}


# Pairwise distance ratios are invariant to translation and global scale. Work
# in a normalized difference space to avoid underflow/overflow before squaring.
.dgpage_squared_distances <- function(X) {
  n <- nrow(X)
  data_scale <- max(abs(X))
  if (data_scale == 0) {
    return(matrix(0, n, n))
  }

  X_unit <- X / data_scale
  X_difference <- sweep(X_unit, 2L, X_unit[1L, ], FUN = "-")
  difference_scale <- max(abs(X_difference))
  if (difference_scale == 0) {
    return(matrix(0, n, n))
  }

  unname(as.matrix(dist(X_difference / difference_scale, method = "euclidean")))^2
}


.dgpage_safe_row_ratio <- function(numerator, denominator) {
  denominator_safe <- denominator
  denominator_safe[denominator_safe == 0] <- 1
  numerator / denominator_safe
}


# -------------------------------------------------------------------
#  DGPAGE as a discriminant_projector extension
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
#' @param X A numeric matrix of size n x d (training samples).
#' @param y A factor (or coercible to factor) of length n (class labels).
#' @param S The n x n similarity graph. If \code{NULL}, we compute it from \code{X,y}.
#' @param D The n x n diversity graph. If \code{NULL}, we compute it from \code{X,y}.
#' @param r The target dimension (number of discriminant directions).
#' @param alpha,beta Hyperparameters in Eq. (12). \code{alpha} must be positive,
#'   \code{beta} must be non-negative, and \code{alpha + beta < 1}.
#' @param maxiter,tol,verbose,seed,ridge,rank_tol Passed to
#'   \code{\link{dgpage_fit}}.
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
#'     \item \code{obj_trace}: objective value after each iteration
#'     \item \code{converged}: whether the relative objective change was below \code{tol}
#'     \item \code{final_rel_change}: final relative objective change
#'     \item \code{optimization_rank}: numerical row-space rank used by the solver
#'     \item \code{requested_r,effective_r}: requested and fitted dimensions
#'     \item \code{ridge,ridge_effective}: requested multiplier and normalized
#'       row-space ridge
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
                                verbose= FALSE,
                                seed   = NULL,
                                q      = 1.5,
                                ridge  = 1e-8,
                                rank_tol = NULL)
{
  X <- as.matrix(X)
  y <- as.factor(y)
  n <- nrow(X)
  if (length(y) != n) {
    stop("Length of y must match nrow(X).")
  }
  assertthat::assert_that(!anyNA(y), msg = "y must not contain missing values.")
  assertthat::assert_that(nlevels(droplevels(y)) >= 2L,
                          msg = "y must contain at least two classes.")

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
                       verbose=verbose, seed=seed,
                       ridge=ridge, rank_tol=rank_tol)

  # Build training scores: s = X %*% P
  # fitres$P is d x r
  s_mat <- X %*% fitres$P  # shape (n x r)
  # sdev => stdev of each column
  sdev_vec <- apply(s_mat, 2, sd)

  # Create a discriminant_projector object with an extra class "dgpage_projector"
  dp <- multivarious::discriminant_projector(
    v       = fitres$P,     # loadings
    s       = s_mat,        # training scores
    sdev    = sdev_vec,     # stdev
    preproc = multivarious::fit(multivarious::pass(), X),
    labels  = y,            # the training labels
    classes = "dgpage_projector"
  )

  # Store additional fields from the fit:
  dp$W         <- fitres$W
  dp$objective <- fitres$objective
  dp$obj_trace <- fitres$obj_trace
  dp$alpha     <- alpha
  dp$beta      <- beta
  dp$iter      <- fitres$iter
  dp$converged <- fitres$converged
  dp$final_rel_change <- fitres$final_rel_change
  dp$initial_objective <- fitres$initial_objective
  dp$ridge <- fitres$ridge
  dp$ridge_effective <- fitres$ridge_effective
  dp$rank_tol <- fitres$rank_tol
  dp$optimization_rank <- fitres$optimization_rank
  dp$requested_r <- fitres$requested_r
  dp$effective_r <- fitres$effective_r
  dp$data_scale <- fitres$data_scale

  # Stored for debugging / prediction:
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
#' uses \pkg{Rnanoflann} for k-nearest-neighbor classification in the projected space.
#'
#' Otherwise, if \code{method} is \code{"lda"} or \code{"euclid"}, we fall back to the
#' standard \code{multivarious::predict.discriminant_projector()} method for linear discriminant
#' or Euclidean nearest-mean classification in subspace.
#'
#' @param object A \code{dgpage_projector} object.
#' @param new_data A numeric matrix (m x d).
#' @param method One of \code{c("1nn","lda","euclid")}. Default \code{"1nn"}.
#' @param type If \code{method="1nn"}, only \code{"class"} is supported and
#'   other values produce an error.
#'   If \code{method \%in\% c("lda","euclid")}, we can do \code{"class"} or \code{"prob"}.
#' @param k Number of neighbors if using \code{method="1nn"}. Default 1. For
#'   \code{k > 1} a majority vote over the k nearest training samples is taken,
#'   with ties broken by factor level order.
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
#' @seealso \code{multivarious::predict.discriminant_projector()}
#' @export
predict.dgpage_projector <- function(object,
                                     new_data,
                                     method = c("1nn","lda","euclid"),
                                     type   = c("class","prob"),
                                     k      = 1,
                                     ...) {
  method <- match.arg(method)
  type <- match.arg(type)

  if (method %in% c("lda","euclid")) {
    # Let the base 'predict.discriminant_projector' do the job.
    object_base <- object
    class(object_base) <- setdiff(class(object_base), "dgpage_projector")
    return(predict(object_base,
                   new_data = new_data,
                   method = method,
                   type = type,
                   ...))
  }

  if (type != "class") {
    stop("method = \"1nn\" supports only type = \"class\".", call. = FALSE)
  }

  # method="1nn": k-NN in the projected space via Rnanoflann.
  # 1) Project new_data into subspace => new_data_proj (m x r)
  new_data <- as.matrix(new_data)
  assertthat::assert_that(is.numeric(new_data), msg = "new_data must be numeric.")
  assertthat::assert_that(all(is.finite(new_data)), msg = "new_data must contain only finite values.")
  assertthat::assert_that(ncol(new_data) == nrow(object$v),
                          msg = "new_data must have the same number of columns as the training data.")
  assertthat::assert_that(length(object$labels) == nrow(object$s),
                          msg = "Length of labels must match the number of training scores.")
  k <- .validate_knn_k(k, nrow(object$s))
  new_data_processed <- multivarious::transform(object$preproc, new_data)
  new_data_proj <- new_data_processed %*% object$v

  # 2) Nearest neighbors among the training scores.
  #    trans = TRUE returns indices as an m x k matrix (verified empirically).
  res <- nn(data   = object$s,
            points = new_data_proj,
            k      = k,
            method = "euclidean",
            trans  = TRUE)

  .knn_vote(object$labels, res$indices)
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
#' @param y A length-\eqn{n} vector/factor of class labels. Each class must have
#'   at least two members.
#' @param q A positive exponent (see Eqs. (5) and (6)). Typical values are between 1 and 3.
#'
#' @return An \eqn{n \times n} matrix \eqn{S}.
#' @export
compute_similarity_graph <- function(X, y, q = 1.5) {
  X <- as.matrix(X)
  assertthat::assert_that(is.numeric(X), msg = "X must be numeric.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(nrow(X) >= 2L && ncol(X) >= 1L,
                          msg = "X must have at least two rows and one column.")
  assertthat::assert_that(.dgpage_is_finite_scalar(q) && q > 0,
                          msg = "q must be a positive finite number.")
  n <- nrow(X)
  y <- .validate_dgpage_labels(y, n)

  distMat <- .dgpage_squared_distances(X)

  # Class-match indicator and per-sample intra/inter counts
  same <- outer(as.integer(y), as.integer(y), "==")
  n_plus <- tabulate(y)[as.integer(y)]
  n_minus <- n - n_plus

  # tau_i^+ = (1 / n^+(c_i)^q) * sum_{l: same class} d_il^2  (self included, d_ii = 0)
  # tau_i^- = (1 / n^-(c_i)^q) * sum_{l: other class} d_il^2
  s_plus <- rowSums(distMat * same)
  s_minus <- rowSums(distMat * (!same))
  tau_plus <- s_plus / n_plus^q
  tau_minus <- s_minus / n_minus^q

  # E_plus[i, j] = exp(-d_ij^2 / tau_i^+): `distMat / tau` recycles tau down
  # columns, i.e. divides row i by tau[i]. Fill only the entries each graph
  # branch uses, avoiding irrelevant overflow/underflow in degenerate cases.
  E_plus <- matrix(0, nrow = n, ncol = n)
  E_minus <- matrix(0, nrow = n, ncol = n)
  E_plus[same] <- exp(-.dgpage_safe_row_ratio(distMat, tau_plus)[same])
  E_minus[!same] <- exp(-.dgpage_safe_row_ratio(distMat, tau_minus)[!same])

  S <- 0.5 * (E_plus + t(E_plus)) * same - 0.5 * (E_minus + t(E_minus)) * (!same)
  diag(S) <- 0
  S
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
#' @param y A length-\eqn{n} vector/factor of class labels. Each class must have
#'   at least two members.
#'
#' @return An \eqn{n \times n} matrix \eqn{D}.
#' @export
compute_diversity_graph <- function(X, y) {
  X <- as.matrix(X)
  assertthat::assert_that(is.numeric(X), msg = "X must be numeric.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(nrow(X) >= 2L && ncol(X) >= 1L,
                          msg = "X must have at least two rows and one column.")
  n <- nrow(X)
  y <- .validate_dgpage_labels(y, n)

  distMat <- .dgpage_squared_distances(X)

  same <- outer(as.integer(y), as.integer(y), "==")

  # rho_i^+, rho_i^- ~ same as tau_i^+, tau_i^- but q=0, i.e. plain sums
  rho_plus <- rowSums(distMat * same)
  rho_minus <- rowSums(distMat * (!same))

  # Note the positive exponent (diversity grows with distance). Fill only the
  # entries each graph branch uses; otherwise zero-distance classes can create
  # Inf * 0 = NaN for irrelevant masked entries.
  E_plus <- matrix(0, nrow = n, ncol = n)
  E_minus <- matrix(0, nrow = n, ncol = n)
  E_plus[same] <- exp(.dgpage_safe_row_ratio(distMat, rho_plus)[same])
  E_minus[!same] <- exp(.dgpage_safe_row_ratio(distMat, rho_minus)[!same])

  D <- 0.5 * (E_plus + t(E_plus)) * same + 0.5 * (E_minus + t(E_minus)) * (!same)
  diag(D) <- 0
  D
}


.prepare_dgpage_row_space <- function(X, rank_tol = NULL) {
  data_scale <- max(abs(X))
  assertthat::assert_that(
    is.finite(data_scale) && data_scale > 0,
    msg = "X must contain non-zero variation."
  )

  X_scaled <- X / data_scale
  sv <- svd(X_scaled, nu = 0L, nv = min(dim(X_scaled)))
  relative_rank_tol <- if (is.null(rank_tol)) {
    max(dim(X_scaled)) * .Machine$double.eps
  } else {
    assertthat::assert_that(
      .dgpage_is_finite_scalar(rank_tol) && rank_tol >= 0 && rank_tol < 1,
      msg = "rank_tol must be NULL or a finite numeric scalar in [0, 1)."
    )
    rank_tol
  }
  threshold <- relative_rank_tol * max(sv$d)
  optimization_rank <- sum(sv$d > threshold)
  assertthat::assert_that(
    optimization_rank >= 1L,
    msg = "X has numerical rank zero at the requested rank tolerance."
  )

  basis <- sv$v[, seq_len(optimization_rank), drop = FALSE]
  X_reduced <- X_scaled %*% basis
  list(
    data_scale = data_scale,
    basis = basis,
    X_reduced = X_reduced,
    singular_values = sv$d[seq_len(optimization_rank)],
    optimization_rank = optimization_rank,
    rank_threshold = threshold,
    relative_rank_tol = relative_rank_tol
  )
}


.update_dgpage_W <- function(Y, DtD, alpha, beta) {
  YtY <- crossprod(Y)
  col_norms <- colSums(Y^2)
  E <- outer(col_norms, col_norms, "+") - 2 * YtY
  H <- YtY + alpha * DtD
  H <- 0.5 * (H + t(H))
  chol_H <- tryCatch(chol(H), error = function(e) NULL)
  if (is.null(chol_H)) {
    stop(
      "The W subproblem is not positive definite. Increase alpha or supply a diversity graph D with sufficient rank.",
      call. = FALSE
    )
  }

  rhs <- YtY - (beta / 2) * E
  backsolve(chol_H, forwardsolve(t(chol_H), rhs))
}


.update_dgpage_P <- function(W, Z, N_inv_half, G_S, alpha, beta, r, I_n) {
  G_W <- .pairwise_laplacian(W)
  K <- tcrossprod(I_n - W)
  A <- K + beta * G_W + (1 - alpha - beta) * G_S
  ZA <- Z %*% A
  M <- tcrossprod(ZA, Z)
  M <- 0.5 * (M + t(M))

  whitened <- crossprod(N_inv_half, M %*% N_inv_half)
  whitened <- 0.5 * (whitened + t(whitened))
  eig <- eigen(whitened, symmetric = TRUE)
  selected <- order(eig$values)[seq_len(r)]
  N_inv_half %*% eig$vectors[, selected, drop = FALSE]
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
#' 1. Compute the numerical row space of \code{X} and initialize \eqn{P}
#'    randomly under the generalized orthonormality constraint.
#' 2. For each iteration:
#'    \itemize{
#'      \item Update \eqn{W} via Eq. (18) (closed-form).
#'      \item Update \eqn{P} by solving the generalized eigenvalue problem in Eq. (23).
#'    }
#' 3. Repeat until convergence or \code{maxiter} is reached.
#'
#' Both alternating steps minimize the same objective
#' \deqn{
#'   F(P, W) = \|P^\top Z - P^\top Z W\|_F^2 + \alpha \|D W\|_F^2
#'           + \beta \sum_{ij} w_{ij} \|P^\top z_i - P^\top z_j\|^2
#'           + (1-\alpha-\beta) \sum_{ij} s_{ij} \|P^\top z_i - P^\top z_j\|^2,
#' }
#' so the objective is non-increasing across iterations (after the first
#' P-update); this is checked by the unit tests.
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
#' @param alpha A strictly positive hyper-parameter in Eq. (12). Typical range:
#'   \eqn{[1e-6, 1e-1]}. Strict positivity, together with a sufficiently ranked
#'   diversity graph, makes the adaptive-graph subproblem bounded.
#' @param beta The hyper-parameter in Eq. (12). Typical range: \eqn{[1e-6, 1e-1]}.
#'   \code{beta} must be non-negative and \code{alpha + beta < 1}.
#' @param maxiter Maximum number of iterations for the alternating updates (>= 1).
#' @param tol Convergence tolerance on the relative change of the objective value.
#' @param verbose Whether to print progress information (via \code{message}).
#' @param seed Optional integer seed for the random initialization of \eqn{P}.
#'   When supplied, the caller's RNG state is saved and restored, so the global
#'   random stream is unaffected. When \code{NULL} (default), the current RNG
#'   stream is used (and advanced) as usual.
#' @param ridge Relative ridge added to the row-space total-scatter matrix before
#'   the generalized eigen-solve, as \code{ridge * mean(diag(N))}. Default
#'   \code{1e-8}. No absolute ridge is added, preserving unit equivariance.
#' @param rank_tol Optional relative singular-value threshold used to determine
#'   the numerical row-space rank of \code{X}. The default is
#'   \code{max(dim(X)) * .Machine$double.eps}.
#'
#' @return A list with:
#' \itemize{
#'   \item \code{P}: the \eqn{d \times effective_r} projection matrix mapped
#'         back from the numerical row space,
#'   \item \code{W}: the \eqn{n \times n} learned adjacency matrix (adaptive graph),
#'   \item \code{objective}: the final objective value,
#'   \item \code{obj_trace}: the objective value after each iteration,
#'   \item \code{iter}: number of iterations taken,
#'   \item \code{converged}: whether the relative objective change fell below \code{tol},
#'   \item \code{final_rel_change}: relative change at the last iteration,
#'   \item \code{requested_r}, \code{effective_r}, and \code{optimization_rank}:
#'         dimension/rank diagnostics,
#'   \item \code{ridge} and \code{ridge_effective}: the requested multiplier and
#'         effective ridge in normalized row-space coordinates,
#'   \item \code{Z}: original training data as \code{t(X)}, retained for prediction.
#' }
#' @export
dgpage_fit <- function(X, S, D, r = 10,
                       alpha = 1e-3, beta = 1e-5,
                       maxiter = 20, tol = 1e-5,
                       verbose = FALSE, seed = NULL,
                       ridge = 1e-8, rank_tol = NULL) {
  X <- as.matrix(X)
  assertthat::assert_that(is.numeric(X), msg = "X must be numeric.")
  assertthat::assert_that(all(is.finite(X)), msg = "X must contain only finite values.")
  assertthat::assert_that(nrow(X) >= 2L && ncol(X) >= 1L,
                          msg = "X must have at least two rows and one column.")
  n <- nrow(X)

  assertthat::assert_that(is.matrix(S) && is.numeric(S) && all(dim(S) == c(n, n)),
                          msg = "S must be an n x n matrix with numeric entries.")
  assertthat::assert_that(is.matrix(D) && is.numeric(D) && all(dim(D) == c(n, n)),
                          msg = "D must be an n x n matrix with numeric entries.")
  assertthat::assert_that(all(is.finite(S)), msg = "S must contain only finite values.")
  assertthat::assert_that(all(is.finite(D)), msg = "D must contain only finite values.")
  assertthat::assert_that(.dgpage_is_positive_integer(r),
                          msg = "r must be a finite positive integer.")
  assertthat::assert_that(.dgpage_is_finite_scalar(alpha) && alpha > 0,
                          msg = "alpha must be a finite positive numeric scalar.")
  assertthat::assert_that(.dgpage_is_finite_scalar(beta) && beta >= 0,
                          msg = "beta must be a finite non-negative numeric scalar.")
  assertthat::assert_that(alpha + beta < 1,
                          msg = "alpha and beta must satisfy alpha + beta < 1.")
  assertthat::assert_that(.dgpage_is_positive_integer(maxiter),
                          msg = "maxiter must be a finite positive integer.")
  assertthat::assert_that(.dgpage_is_finite_scalar(tol) && tol > 0,
                          msg = "tol must be a finite positive numeric scalar.")
  assertthat::assert_that(.dgpage_is_finite_scalar(ridge) && ridge >= 0,
                          msg = "ridge must be a finite non-negative numeric scalar.")
  assertthat::assert_that(is.logical(verbose) && length(verbose) == 1L && !is.na(verbose),
                          msg = "verbose must be TRUE or FALSE.")
  assertthat::assert_that(
    is.null(seed) || (.dgpage_is_finite_scalar(seed) && seed >= 0 && seed == round(seed)),
    msg = "seed must be NULL or a finite integer."
  )

  row_space <- .prepare_dgpage_row_space(X, rank_tol)
  requested_r <- as.integer(r)
  if (requested_r > row_space$optimization_rank) {
    warning(
      "r cannot exceed the numerical rank of X (", row_space$optimization_rank,
      "). Setting r = ", row_space$optimization_rank, "."
    )
    r <- row_space$optimization_rank
  } else {
    r <- requested_r
  }

  # Optional reproducible initialization that does not clobber the caller's
  # RNG state.
  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      old_seed <- get(".Random.seed", envir = globalenv())
      on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
    } else {
      on.exit(rm(".Random.seed", envir = globalenv()), add = TRUE)
    }
    set.seed(seed)
  }

  # Optimize only in the numerical row space of X. This removes feature-null
  # directions and reduces each eigenproblem from d to optimization_rank.
  X_reduced <- row_space$X_reduced
  Z_reduced <- t(X_reduced)
  optimization_rank <- row_space$optimization_rank
  I_n <- diag(1, n)
  DtD <- crossprod(D)
  G_S <- .pairwise_laplacian(S)
  N <- tcrossprod(Z_reduced)
  ridge_eff <- ridge * mean(diag(N))
  N_reg <- N + ridge_eff * diag(optimization_rank)
  chol_N <- tryCatch(chol(N_reg), error = function(e) NULL)
  if (is.null(chol_N)) {
    stop("The row-space total-scatter matrix is not positive definite; increase ridge or rank_tol.",
         call. = FALSE)
  }
  N_inv_half <- backsolve(chol_N, diag(optimization_rank))

  # Objective F(P, W) from Eq. (12), used for monitoring and convergence.
  # tr(Y G Y^T) computed as sum((Y %*% G) * Y) to avoid forming d x d products.
  graph_quad <- function(Y, G) sum((Y %*% G) * Y)
  compute_objective <- function(Y, W, G_W) {
    part1 <- sum((Y - Y %*% W)^2)                 # ||P^T Z (I - W)||_F^2
    part2 <- alpha * sum((D %*% W)^2)             # alpha ||D W||_F^2
    part3 <- beta * graph_quad(Y, G_W) +          # beta sum_ij w_ij e_ij
      (1 - alpha - beta) * graph_quad(Y, G_S)     # (1-a-b) sum_ij s_ij e_ij
    part1 + part2 + part3
  }

  # Initialize in whitened coordinates so P satisfies the generalized
  # constraint before the first W update, independent of measurement units.
  tmp <- matrix(rnorm(optimization_rank * r), nrow = optimization_rank, ncol = r)
  P_reduced <- N_inv_half %*% qr.Q(qr(tmp))
  Y <- crossprod(P_reduced, Z_reduced)
  W <- .update_dgpage_W(Y, DtD, alpha, beta)
  old_obj <- compute_objective(Y, W, .pairwise_laplacian(W))
  initial_objective <- old_obj
  if (verbose) message(sprintf("Initial objective = %.6f", old_obj))

  obj_trace <- numeric(maxiter)
  converged <- FALSE
  final_rel_change <- Inf

  for (iter in seq_len(maxiter)) {
    # End each iteration after W is optimized for the current P, so every
    # returned state is coherent even when maxiter is exhausted.
    P_reduced <- .update_dgpage_P(
      W, Z_reduced, N_inv_half, G_S, alpha, beta, r, I_n
    )
    Y <- crossprod(P_reduced, Z_reduced)
    W <- .update_dgpage_W(Y, DtD, alpha, beta)
    new_obj <- compute_objective(Y, W, .pairwise_laplacian(W))
    assertthat::assert_that(is.finite(new_obj), msg = "DGPAGE objective became non-finite.")
    obj_trace[iter] <- new_obj
    final_rel_change <- abs(new_obj - old_obj) / max(abs(old_obj), .Machine$double.eps)
    if (verbose) {
      message(sprintf("Iter %2d: objective=%.6f, rel_diff=%.6g", iter, new_obj, final_rel_change))
    }
    old_obj <- new_obj

    if (final_rel_change < tol) {
      converged <- TRUE
      if (verbose) message("Converged.")
      break
    }
  }

  obj_trace <- obj_trace[seq_len(iter)]
  P <- (row_space$basis %*% P_reduced) / row_space$data_scale

  list(
    P = P,
    P_reduced = P_reduced,
    W = W,
    objective = old_obj,
    obj_trace = obj_trace,
    iter = iter,
    converged = converged,
    final_rel_change = final_rel_change,
    initial_objective = initial_objective,
    Z = t(X),
    Z_reduced = Z_reduced,
    row_basis = row_space$basis,
    optimization_rank = optimization_rank,
    requested_r = requested_r,
    effective_r = r,
    rank_threshold = row_space$rank_threshold,
    rank_tol = row_space$relative_rank_tol,
    data_scale = row_space$data_scale,
    ridge = ridge,
    ridge_effective = ridge_eff
  )
}


# -------------------------------------------------------------------------
#  3) Prediction (k-NN) using Rnanoflann
# -------------------------------------------------------------------------

#' Predict labels using a trained DGPAGE model via k-NN classification
#'
#' After you have fit the DGPAGE model to some labeled training data, store the
#' training labels on the model (\code{model$ytrain <- y}). For \emph{new} data
#' \code{Xtest}, this function projects both training and test data using the
#' learned \code{P} and performs a k-nearest-neighbor search in the
#' low-dimensional subspace using \pkg{Rnanoflann}.
#'
#' @param model The list returned by \code{\link{dgpage_fit}}, augmented with
#'   \code{ytrain} (the vector of training labels). The training embedding is
#'   derived from \code{model$P} and \code{model$Z}, so nothing else needs to
#'   be stored.
#' @param Xtest A numeric matrix of size \eqn{m \times d} (test data).
#' @param k Number of neighbors (default 1). For \code{k > 1} a majority vote
#'   is taken, with ties broken by factor level order.
#' @param method Distance method for \code{\link[Rnanoflann]{nn}}. Default "euclidean".
#'
#' @return A factor of length \eqn{m} with predicted labels.
#' @export
#' @seealso \code{\link[Rnanoflann]{nn}}
#' @examples
#' \dontrun{
#'   model <- dgpage_fit(Xtrain, S, D, r = 2)
#'   model$ytrain <- ytrain
#'   preds <- dgpage_predict(model, Xtest)
#' }
dgpage_predict <- function(model, Xtest, k=1, method="euclidean") {
  if (is.null(model$P) || is.null(model$Z)) {
    stop("model does not contain P or Z. Did you remove them?")
  }
  if (is.null(model$ytrain)) {
    stop("model does not contain ytrain (the training labels). ",
         "Store them after training, e.g. model$ytrain <- y.")
  }
  Xtest <- as.matrix(Xtest)
  assertthat::assert_that(is.numeric(Xtest), msg = "Xtest must be numeric.")
  assertthat::assert_that(all(is.finite(Xtest)), msg = "Xtest must contain only finite values.")
  assertthat::assert_that(ncol(Xtest) == nrow(model$P),
                          msg = "Xtest must have the same number of columns as the training data.")
  assertthat::assert_that(length(model$ytrain) == ncol(model$Z),
                          msg = "Length of ytrain must match the number of training samples.")
  k <- .validate_knn_k(k, ncol(model$Z))

  # Training embedding (n x r), derived from the stored P and Z
  Ytrain <- crossprod(model$Z, model$P)
  # Test embedding (m x r)
  Ytest <- Xtest %*% model$P

  # trans = TRUE returns indices as an m x k matrix (verified empirically)
  res <- nn(
    data   = Ytrain,
    points = Ytest,
    k      = k,
    method = method,
    trans  = TRUE
  )

  .knn_vote(model$ytrain, res$indices)
}

#' @title Print method for dgpage_projector objects
#' @description Print a \code{dgpage_projector} object and summary information.
#' @param x A \code{dgpage_projector} object.
#' @param ... Additional arguments passed to the next print method.
#' @method print dgpage_projector
#' @export
print.dgpage_projector <- function(x, ...) {
  cat("A 'dgpage_projector' object (subclass of 'discriminant_projector').\n")
  cat("Learned adjacency graph W of size:", nrow(x$W), "x", ncol(x$W), "\n")
  convergence_label <- if (isTRUE(x$converged)) "converged" else "not converged"
  cat(sprintf("Objective: %.3f (%d iterations; %s)\n",
              x$objective, x$iter, convergence_label))
  cat("Embedding dimensions:", x$effective_r, "of", x$requested_r,
      "requested; numerical rank:", x$optimization_rank, "\n")
  cat("alpha:", x$alpha, " beta:", x$beta, "\n")
  # Then delegate the rest to the standard print for discriminant_projector:
  NextMethod()  # calls print.discriminant_projector
}
