# moving_scatter <- function(X, window) {
#   window <- as.integer(window)
#   assertthat::assert_that(window < ncol(X))
#   assertthat::assert_that(window >= 3)
#   assertthat::assert_that(window %% 2 != 0, msg="`window` must be an odd number >= 3")
#   
#   mu <- colMeans(X)
#   S <- NULL
#   halfwindow <- floor(window/2)
#   #nwin <- ncol(X) - window
#   for (i in seq(halfwindow+1,ncol(X)-halfwindow)) {
#     ## i min half-window, i + half_window, window should be odd.
#     ind <- seq(i-halfwindow, i+halfwindow)
#     print(ind)
#     if (is.null(S)) {
#       S <- total_scatter(X[,ind], mu[ind])
#     } else {
#       S <- S + total_scatter(X[,ind], mu[ind])
#     }
#   }
#   
#   S <- S/length(seq(halfwindow+1,nwin-halfwindow))
#   cols <- rep(1:7, 7)
#   rows <- rep(1:7, each=7)
#   
# }

#' @keywords internal
pooled_scatter <- function(X, Y) {
  ina <- as.integer(droplevels(Y))
  s <- crossprod(X)
  ni <- sqrt(tabulate(ina))
  mi <- rowsum(X, ina)/ni
  k <- length(ni)
  denom <- dim(X)[1] - k
  for (i in 1:k) s <- s - tcrossprod(mi[i, ])
  s
}

#' @keywords internal
total_scatter <- function(X, mu) {
  p <- ncol(X)
  St <- array(0, c(p,p))
  for (i in 1:nrow(X)) {
    delta <- X[i,] - mu
    St <- St + outer(delta,delta)
  }
  
  St
}

#' @keywords internal
between_class_scatter <- function(X, Y, mu) {
  p <- ncol(X)
  Y <- droplevels(as.factor(Y))
  levs <- levels(Y)
  
  gmeans <- multivarious::group_means(Y,X)
  gmeans <- sweep(gmeans, 2, mu, "-")
  
  n <- tabulate(Y)
  
  res <- lapply(seq_along(levs), function(i) {
    n[i] * tcrossprod(gmeans[i,], gmeans[i,])
  })
  
  Reduce("+", res)
  
}

#' @keywords internal
within_class_scatter <- function(X, Y) {
  pooled_scatter(X,Y)
}


binary_between_scatter <-function(X, id1, id2) {
  gmean <- colMeans(X[c(id1,id2),])
  x1 <- length(id1) + tcrossprod(colMeans(X[id1,]) - gmean)
  x2 <- length(id2) + tcrossprod(colMeans(X[id2,]) - gmean)
  ## x1 == x2, so do we ned to calculate both? x1*2
  x1+x2
  
}

pairwise_within_scatter <- function(X, labels, l1, l2) {
  
  pairs <- combn(unique(as.character(labels)),2)
  
  lapply(1:ncol(pairs), function(i) {
    print(i)
    ids <- which(labels %in% pairs[,i])
    w <- pooled_scatter(X[ids,], labels[ids])
    
    b <- binary_between_scatter(X, ids[labels[ids] == pairs[1,i]], ids[labels[ids] == pairs[2,i]])
    list(w=w, b=b)
  })
  
  
  
}