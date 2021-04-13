
#' uncorrelated linear discriminant analysis
#' 
#' @param Y the class labels
#' @param X the data matrix
#' @param preproc the pre-processing function
#' 
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' Y <- iris[,5]
#' 
#' res <- ulda(X,Y)
#' @export
ulda <- function(X, Y,preproc=center(), mu=0, tol=1e-6) {
  Y <- as.factor(Y)
  
  procres <- prep(preproc)
  Xp <- init_transform(procres, X)
  
 
  levs <- levels(Y)
  nc <- length(levs)
  
  cprobs <- table(Y)/length(Y)
  cmeans <- group_means(Y,Xp)
  gmean <- colMeans(Xp)
  
  Hb <- do.call(cbind, lapply(1:nc, function(i) {
    sqrt(cprobs[i]) * (cmeans[i,] - gmean)
  }))
  
  Ht <- t(sweep(Xp, 2, gmean, "-"))
  t <- min(dim(Ht))
  
  svd_ht <- svd(Ht)
  keep <- which(svd_ht$d > tol)
  
  # B <- diag(svd_ht$d[keep]) %*% t(svd_ht$u[,keep]) %*% Hb
  B <- diag(1/svd_ht$d[keep]) %*% t(svd_ht$u[,keep]) %*% Hb
  
  svd_B <- svd(B)
  keep_b <- which(svd_B$d > tol)
  #q <- min(dim(B))
  
  vecs <- svd_ht$u[,keep,drop=FALSE] %*% diag(1/svd_ht$d[keep,drop=FALSE]) %*% svd_B$u[,keep_b]
  
  ## regularization here
  #if (mu > 0) {
  #  B <- diag(svd_B$d[keep_b]^2)
  #  W <- 
  #}
  
  multivarious::discriminant_projector(v=vecs, preproc=procres, labels=Y, classes="ulda")
  
}

