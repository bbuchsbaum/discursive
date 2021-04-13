
weighted_group_means <- function(X, F) {
  ret <- do.call(rbind, lapply(1:nrow(F), function(i) {
    w <- F[i,]
    matrixStats::colWeightedMeans(X, w/sum(w))
  }))
  
  row.names(ret) <- row.names(F)
  ret
}


##A general soft label based Linear Discriminant Analysis for
##semi-supervised dimensionality reduction
## https://paperpile.com/app/p/03eb8cb3-2326-0cc5-a440-02c6aa543bf3 

## see also, maybe similar:
## A weighted linear discriminant analysis framework for multi-label
## feature extraction

#' @param X the data matrix
#' @param C the class weight matrix `nrow(C)` == `nrow(X)` and `ncol(C)` is equal to number of class labels.
soft_lda <- function(X, C, preproc=pass(), dp=min(dim(X)), di=dp-1, dl=ncol(C)-1) {
  assert_that(nrow(C) == nrow(X))
  
  if (is.null(colnames(C))) {
    colnames(C) <- paste0("c_", 1:ncol(C))
  }
  
  chk::chk_true(all(C >= 0), msg="all weights in 'C' must be positive")
  
  ## pre-process X 
  procres <- prep(preproc)
  Xp <- multivarious::init_transform(procres, X)
  
 
  ## row sums of soft-label matrix
  E <- Matrix::Diagonal(x=rowSums(C))
  
  ## column sums of soft-label matrix
  ## this is essentially the variable weights (could be pre-standardized)
  G <- diag(colSums(C))
  
  ## transposed weight matrix: each row is a weight vector for a class
  F <- t(as.matrix(C))
  
  ## FtGF is therefore the weeighted covariance of the class probabilities
  FtGF <- (t(F) %*% diag(1/diag(G)) %*% F)
  
  sw_scatter <- function(X) {
    Xt <- t(X)
    t(X) %*% (E - FtGF) %*% X
  }
  
  sb_scatter <- function(X) {
    Xt <- t(X)
    #e <- matrix(e)
    #num <- E %*% e %*% t(e) %*% E
    num <- tcrossprod(diag(E), rep(1,ncol(E))) %*% E
    #denom <- t(e) %*% E %*% e
    denom <- sum(diag(E))
    M <- FtGF - num/denom
    ##Xt %*% M %*% t(Xt)
    crossprod(X, (M %*% X))
  }
  
  ## reduce Xp into dp dimensions
  pca_red <- multivarious::pca(Xp, ncomp=dp, preproc=center())
  
  ## get projection -- assumes Xp is centered...
  proj_dp <- coef(pca_red)
  
  ## get pca basis
  Xpca <- scores(pca_red)
  
  
  gmeans <- weighted_group_means(Xpca, F)
  mu <- colMeans(gmeans)
  Sw <- sw_scatter(Xpca)
  Sb <- sb_scatter(Xpca)
  
  
  ## this is just pca_lda
  E_i <- RSpectra::eigs_sym(Sw, k=di)
  proj_di <- E_i$vectors %*% diag(1/sqrt(E_i$values))
  
  gmeans_proj <- (gmeans %*% proj_di)
  
  E_l <- pca(gmeans_proj, ncomp=dl)
  
  proj_dl <- loadings(E_l)
  
  proj_final <- proj_dp %*% proj_di %*% proj_dl
  
  projector(procres, ncomp=ncol(proj_final), v=proj_final, classes="sl_lda")
  
  #St <- st_scatter(X, F, mu)
  
}





