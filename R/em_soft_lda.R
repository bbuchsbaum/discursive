#' @title Partially Supervised LDA with Soft Labels (Robust Version)
#'
#' @description Fits a Linear Discriminant Analysis (LDA) model using soft labels
#' and the Evidential EM (E²M) algorithm, with improved robustness and optional regularization.
#'
#' @param X A numeric matrix n x d, the feature matrix.
#' @param PL A numeric matrix n x K of plausibility values for each class and instance.
#' @param max_iter Integer, maximum number of E²M iterations. Default: 100.
#' @param tol Numeric tolerance for convergence in the log-likelihood. Default: 1e-6.
#' @param n_starts Integer, number of random initializations. The best solution (highest final log-likelihood) is chosen. Default: 5.
#' @param reg Numeric, a small ridge penalty to add to the covariance matrix for numerical stability. Default: 1e-9.
#' @param verbose Logical, if TRUE prints progress messages. Default: FALSE.
#'
#' @return A list with:
#'   \item{pi}{Estimated class priors}
#'   \item{mu}{Estimated class means}
#'   \item{Sigma}{Estimated covariance matrix}
#'   \item{zeta}{Posterior class probabilities (n x K)}
#'   \item{loglik}{Final log evidential likelihood}
#'   \item{iter}{Number of iterations performed}
#'
#' @references 
#' Quost, B., Denoeux, T., Li, S. (2017). Parametric classification with soft labels 
#' using the Evidential EM algorithm. Advances in Data Analysis and Classification, 
#' 11(4), 659-690.
#'
#' @examples
#' set.seed(123)
#' n <- 100; d <- 2; K <- 3
#' X <- rbind(
#'   MASS::mvrnorm(n, c(1,0), diag(2)),
#'   MASS::mvrnorm(n, c(-1,1), diag(2)),
#'   MASS::mvrnorm(n, c(0,-1), diag(2))
#' )
#' Y <- c(rep(1,n), rep(2,n), rep(3,n))
#' # Soft labels: add uncertainty
#' PL <- matrix(0, 3*n, K)
#' for (i in 1:(3*n)) {
#'   if (runif(1)<0.2) {
#'     alt <- sample(setdiff(1:K, Y[i]),1)
#'     PL[i,Y[i]] <- 0.5
#'     PL[i,alt] <- 0.5
#'   } else {
#'     PL[i,Y[i]] <- 1
#'   }
#' }
#' res <- em_soft_lda(X, PL, verbose=TRUE)
#' str(res)
em_soft_lda <- function(X, PL, max_iter=100, tol=1e-6, n_starts=5, reg=1e-9, verbose=FALSE) {
  # Basic checks
  if(!is.matrix(X)) stop("X must be a matrix.")
  if(!is.matrix(PL)) stop("PL must be a matrix.")
  n <- nrow(X); d <- ncol(X)
  if(nrow(PL)!=n) stop("PL must have same number of rows as X.")
  K <- ncol(PL)
  if(K < 2) stop("At least two classes are required.")
  if(any(PL<0)) stop("PL cannot contain negative values.")
  if(any(rowSums(PL)==0)) stop("Each instance should have at least one positive plausibility.")
  
  gauss_den <- function(x, mu, Sigma_inv, logdetS) {
    # Given precomputed Sigma_inv and logdetS
    # x: n x d, mu: 1 x d
    # returns vector of densities
    diff <- x - matrix(mu, nrow(x), d, byrow=TRUE)
    d2 <- rowSums((diff %*% Sigma_inv)*diff)
    out <- exp(-0.5*d2)/( (2*pi)^(d/2)*exp(logdetS/2))
    out
  }
  
  best_res <- NULL
  best_ll <- -Inf
  
  # multiple starts
  for (start_id in 1:n_starts) {
    # Initialization:
    # k-means for initialization
    cl <- kmeans(X, centers=K, nstart=1)
    pi <- table(cl$cluster)/n
    mu <- cl$centers
    Sigma <- cov(X) + reg*diag(d)
    Sigma_inv <- solve(Sigma)
    logdetS <- as.numeric(determinant(Sigma, logarithm=TRUE)$modulus)
    
    old_ll <- -Inf
    
    for (it in 1:max_iter) {
      # E-step:
      # Compute phi_i(k)
      phi_mat <- matrix(0, n, K)
      for (k in 1:K) {
        phi_mat[,k] <- gauss_den(X, mu[k,], Sigma_inv, logdetS)
      }
      num <- PL * (matrix(pi, n, K, byrow=TRUE)*phi_mat)
      denom_vec <- rowSums(num)
      zeta <- num / denom_vec
      
      # M-step:
      nk <- colSums(zeta)
      pi <- nk/n
      mu <- (t(zeta) %*% X)/nk
      
      # Sigma:
      S <- matrix(0, d, d)
      for (k in 1:K) {
        diff <- X - matrix(mu[k,], n, d, byrow=TRUE)
        # Weighted sum:
        S <- S + t(diff)*zeta[,k] %*% diff
      }
      Sigma <- S/n + reg*diag(d)
      
      # Check numerical stability:
      # If Sigma not invertible, increase reg
      attempt <- 0
      while (TRUE) {
        detS <- determinant(Sigma, logarithm=TRUE)$modulus
        if (is.infinite(detS)) {
          # Increase reg
          attempt <- attempt + 1
          if (attempt>10) stop("Cannot stabilize Sigma.")
          Sigma <- Sigma + reg*diag(d)*10
        } else {
          break
        }
      }
      Sigma_inv <- solve(Sigma)
      logdetS <- as.numeric(determinant(Sigma, logarithm=TRUE)$modulus)
      
      # Compute ll:
      ll <- sum(log(denom_vec))
      if (verbose) cat("Start",start_id,"Iter",it,"LL:",ll,"\n")
      if (abs(ll - old_ll)<tol) {
        old_ll <- ll
        break
      }
      old_ll <- ll
    }
    if (old_ll > best_ll) {
      best_ll <- old_ll
      best_res <- list(pi=pi, mu=mu, Sigma=Sigma, zeta=zeta, loglik=old_ll, iter=it)
    }
  }
  
  best_res
}


#' @title Partially Supervised Logistic Regression with Soft Labels (Robust Version)
#'
#' @description Fits a logistic regression model using E²M with soft labels, adding robustness:
#' multiple initial attempts, ridge penalty, stable line search, and optional verbosity.
#'
#' @param X A numeric matrix n x d.
#' @param PL A numeric matrix n x K of plausibilities.
#' @param max_iter Integer, max E²M iterations. Default 100.
#' @param tol Tolerance for convergence in log-likelihood. Default 1e-6.
#' @param lambda Ridge regularization parameter for coefficients. Default 1e-5.
#' @param n_starts Number of initializations to try. Default 3.
#' @param verbose Logical, if TRUE prints progress. Default FALSE.
#'
#' @return A list:
#'   \item{beta}{(d+1) x (K-1) coefficient matrix}
#'   \item{zeta}{Posterior class probabilities n x K}
#'   \item{loglik}{Final log-likelihood}
#'   \item{iter}{Iterations}
#'
#' @references 
#' Quost, B., Denoeux, T., Li, S. (2017). Parametric classification with soft labels 
#' using the Evidential EM algorithm. Advances in Data Analysis and Classification, 11(4), 659-690.
soft_lr <- function(X, PL, max_iter=100, tol=1e-6, lambda=1e-5, n_starts=3, verbose=FALSE) {
  if(!is.matrix(X)) stop("X must be a matrix.")
  if(!is.matrix(PL)) stop("PL must be a matrix.")
  n <- nrow(X); d <- ncol(X)
  if(nrow(PL)!=n) stop("PL and X must have same number of rows.")
  K <- ncol(PL)
  if(K<2) stop("Need at least two classes for logistic regression.")
  if(any(PL<0)) stop("PL cannot have negative values.")
  if(any(rowSums(PL)==0)) stop("Each instance must have at least one positive plausibility.")
  
  Xtilde <- cbind(1,X)
  
  softmax_prob <- function(eta) {
    # eta: n x (K-1)
    # returns n x K
    denom <- 1+rowSums(exp(eta))
    cbind(exp(eta)/denom, 1/denom)
  }
  
  # Objective: L(theta)=sum_i log sum_k PL_ik p_k(w_i;theta)
  # We try multiple starts:
  best_ll <- -Inf
  best_res <- NULL
  
  for (start_id in 1:n_starts) {
    # Initialize beta
    beta <- matrix(rnorm((d+1)*(K-1), sd=0.1), d+1, K-1)
    
    old_ll <- -Inf
    for (it in 1:max_iter) {
      eta <- Xtilde %*% beta
      Pmat <- softmax_prob(eta)
      
      num <- PL*Pmat
      denom_vec <- rowSums(num)
      zeta <- num/denom_vec
      
      ll <- sum(log(denom_vec))
      if (verbose) cat("Start",start_id,"Iter",it,"LL:",ll,"\n")
      if (abs(ll - old_ll)<tol) {
        old_ll <- ll
        break
      }
      
      # M-step: one Newton step
      Y_star <- zeta[,1:(K-1),drop=FALSE]
      diff_mat <- Y_star - Pmat[,1:(K-1),drop=FALSE]
      
      grad <- t(Xtilde) %*% diff_mat - lambda * cbind(beta)
      # Hessian: block structure
      # H = sum_i x_i^T R_i x_i, 
      # R_i = diag(p_i) - p_i p_i^T for i-th obs (restricted to first K-1 classes)
      d1 <- (d + 1) * (K - 1) # dimension of vectorized beta
      H <- matrix(0, d1, d1)
      # build Hessian:
      for (i in 1:n) {
        p_i <- Pmat[i,1:(K-1)]
        R_i <- diag(p_i)-outer(p_i,p_i)
        Xi <- matrix(Xtilde[i,],d+1,1)
        # Kronecker trick:
        # H = H - (Xi Xi^T) x R_i
        # vectorize approach:
        # more directly:
        H <- H - (Xi %*% t(Xi)) %x% R_i
      }
      # Add ridge
      H <- H - diag(lambda, d1)
      
      vec_beta <- as.vector(beta)
      vec_grad <- as.vector(grad)
      
      # Line search:
      # Newton direction:
      delta <- tryCatch(solve(H, vec_grad), error = function(e) {
        # If solve fails, progressively add more ridge until invertible
        temp_lambda <- lambda * 10
        for (rtry in 1:5) {
          H2 <- H - diag(temp_lambda, d1)
          test <- try(solve(H2, vec_grad), silent = TRUE)
          if (!inherits(test, "try-error")) {
            return(test)
          }
          temp_lambda <- temp_lambda * 10
        }
        stop("Cannot invert Hessian even after increasing ridge.")
      })
      
      step <- 1
      repeat {
        new_vec <- vec_beta + step*delta
        new_beta <- matrix(new_vec, d+1, K-1)
        eta_new <- Xtilde %*% new_beta
        Pmat_new <- softmax_prob(eta_new)
        ll_new <- sum(log(rowSums(PL*Pmat_new)))
        if (ll_new > old_ll || step<1e-12) {
          beta <- new_beta
          old_ll <- ll_new
          break
        } else {
          step <- step/2
        }
      }
    }
    if (old_ll>best_ll) {
      # Compute final zeta
      eta <- Xtilde %*% beta
      Pmat <- softmax_prob(eta)
      num <- PL*Pmat
      denom_vec <- rowSums(num)
      zeta <- num/denom_vec
      best_res <- list(beta=beta, zeta=zeta, loglik=old_ll, iter=it)
      best_ll <- old_ll
    }
  }
  
  best_res
}
