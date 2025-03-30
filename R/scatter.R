moving_scatter <- function(X, window) {
  window <- as.integer(window)
  assertthat::assert_that(window < ncol(X))
  assertthat::assert_that(window >= 3)
  assertthat::assert_that(window %% 2 != 0, msg="`window` must be an odd number >= 3")
  
  halfwindow <- (window - 1) / 2
  # Number of windows we will consider
  # For i in [halfwindow+1, ncol(X)-halfwindow]
  # The count is: (ncol(X)-halfwindow) - (halfwindow+1) + 1 = ncol(X) - 2*halfwindow
  windows_count <- ncol(X) - 2*halfwindow
  
  # Initialize scatter sum
  S <- NULL
  
  for (i in seq(halfwindow + 1, ncol(X) - halfwindow)) {
    # Indices of the current window
    ind <- seq(i - halfwindow, i + halfwindow)
    
    # Compute the mean of just these columns
    mu_window <- colMeans(X[, ind, drop=FALSE])
    
    # Compute the total scatter for this window
    S_window <- total_scatter(X[, ind, drop=FALSE], mu_window)
    
    # Accumulate
    if (is.null(S)) {
      S <- S_window
    } else {
      S <- S + S_window
    }
  }
  
  # Average over all windows
  S <- S / windows_count
  
  return(S)
}

#' Calculate Pooled Scatter (Within-Class Scatter) Matrix
#'
#' Computes the within-class scatter matrix for a dataset with given class labels.
#' This uses a trick involving the class means and their counts to produce:
#' \deqn{S_w = \sum_i x_i x_i^T - \sum_k n_k M_k M_k^T}
#'
#' @param X A numeric matrix (n x d). Rows are samples, columns are features.
#' @param Y A factor or numeric vector of length n representing class labels.
#' @return A numeric (d x d) matrix representing the within-class scatter matrix.
#' @keywords internal
pooled_scatter <- function(X, Y) {
  Y <- droplevels(as.factor(Y))
  ina <- as.integer(Y)
  
  # s = sum_i x_i x_i^T
  s <- crossprod(X)
  
  # ni = sqrt of class counts; tabulate(ina) gives class counts
  ni <- sqrt(tabulate(ina))
  
  # mi[i,] = rowsum(X,ina)/ni = (sum of samples in class i) / sqrt(n_i)
  # tcrossprod(mi[i,]) = (sum of class i samples * sum of class i samples^T)/n_i = n_i * M_k M_k^T
  mi <- rowsum(X, ina) / ni
  
  k <- length(ni)
  
  # Subtract sum_k n_k M_k M_k^T to get within-class scatter
  for (i in 1:k) {
    s <- s - tcrossprod(mi[i, ])
  }
  
  s
}


#' Calculate Total Scatter Matrix
#'
#' Computes the total scatter matrix \eqn{S_t = \sum_i (x_i - \mu)(x_i - \mu)^T}.
#'
#' @param X A numeric matrix (n x d).
#' @param mu A numeric vector of length d representing the overall mean.
#' @return A numeric (d x d) total scatter matrix.
#' @keywords internal
total_scatter <- function(X, mu) {
  # A more efficient R-way: St <- crossprod(sweep(X, 2, mu))
  # But the loop is correct, just less efficient.
  
  p <- ncol(X)
  St <- matrix(0, nrow = p, ncol = p)
  for (i in 1:nrow(X)) {
    delta <- X[i, ] - mu
    St <- St + tcrossprod(delta)
  }
  St
}


#' Calculate Between-Class Scatter Matrix (via C++)
#'
#' Computes the between-class scatter matrix using a C++ function `betweenSScpp`.
#'
#' @param X A numeric matrix (n x d).
#' @param Y A factor or numeric vector of length n.
#' @return A numeric (d x d) between-class scatter matrix.
#' @keywords internal
between_class_scatter <- function(X, Y) {
  ina <- as.integer(droplevels(as.factor(Y)))
  betweenSScpp(X, ina)
}


#' Calculate Within-Class Scatter Matrix (via C++)
#'
#' Computes the within-class scatter matrix using a C++ function `withinSScpp`.
#'
#' @param X A numeric matrix (n x d).
#' @param Y A factor or numeric vector of length n.
#' @return A numeric (d x d) within-class scatter matrix.
#' @keywords internal
within_class_scatter <- function(X, Y) {
  ina <- as.integer(droplevels(as.factor(Y)))
  withinSScpp(X, ina)
}


#' Calculate Binary Between-Class Scatter Matrix
#'
#' For two distinct groups within a dataset, this computes:
#' \deqn{S_b = n_1 (m_1 - m)(m_1 - m)^T + n_2 (m_2 - m)(m_2 - m)^T}
#' where \(m_1\) and \(m_2\) are the class means and \(m\) is the mean of both groups combined.
#'
#' @param X A numeric matrix (n x d).
#' @param id1 A vector of indices for the first group.
#' @param id2 A vector of indices for the second group.
#' @return A numeric (d x d) between-class scatter matrix for the two groups.
#' @keywords internal
binary_between_scatter <- function(X, id1, id2) {
  gmean <- colMeans(X[c(id1, id2), , drop = FALSE])
  
  n1 <- length(id1)
  n2 <- length(id2)
  
  m1 <- colMeans(X[id1, , drop = FALSE])
  m2 <- colMeans(X[id2, , drop = FALSE])
  
  v1 <- (m1 - gmean)
  v2 <- (m2 - gmean)
  
  # Corrected to multiply by n1 and n2 respectively
  x1 <- n1 * tcrossprod(v1)
  x2 <- n2 * tcrossprod(v2)
  
  x1 + x2
}


#' Calculate Pairwise Within-Class and Between-Class Scatter for Class Pairs
#'
#' For each pair of classes in the dataset, this function computes the within-class
#' and between-class scatter matrices restricted to those two classes.
#'
#' @param X A numeric matrix (n x d).
#' @param labels A factor or numeric vector of length n representing class labels.
#' @return A list of length equal to the number of class pairs, where each element is
#'         a list with \code{w} (within-class) and \code{b} (between-class) scatter matrices.
#' @keywords internal
pairwise_within_scatter <- function(X, labels) {
  lbls <- unique(as.character(labels))
  pairs <- combn(lbls, 2)
  
  lapply(1:ncol(pairs), function(i) {
    pair_classes <- pairs[, i]
    ids <- which(labels %in% pair_classes)
    w <- within_class_scatter(X[ids, , drop = FALSE], labels[ids])
    
    id1 <- ids[labels[ids] == pair_classes[1]]
    id2 <- ids[labels[ids] == pair_classes[2]]
    b <- binary_between_scatter(X, id1, id2)
    
    list(w = w, b = b)
  })
}