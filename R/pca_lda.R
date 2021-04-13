
#pca_lda_fit <- function(Xpca, Sb, Sw, dp, di, dl) {}

#' @inheritsParams ulda
#' @param dp the dimension of the initial pca projection
#' @param di the dimension of the within-class projection
#' @param dl the dimension of the between class projection
#' @export
#' @examples 
#' 
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' Y <- iris[,5]
#' res <- pca_lda(X,Y, di=3)
#' 
#' @export
pca_lda <- function(X, Y, preproc=center(), dp=min(dim(X)), di=dp-1, dl=length(unique(Y))-1) {
  chk::chk_range(ncol(X), c(2, 10e6))
  chk::chk_range(nrow(X), c(2, 10e6))
  chk::chk_range(dp, c(2, min(dim(X))))
  chk::chk_range(di, c(2, dp-1))
  chk::chk_range(dl, 1, length(unique(Y))-1)
  
  Y <- as.factor(Y)
  
  procres <- multivarious::prep(preproc)
  
  Xp <- init_transform(procres, X)
  
  pca_basis <- multivarious::pca(Xp, ncomp=dp)

  proj_dp <- pca_basis$v
  
  Xpca <- scores(pca_basis)
  
  Sw <- within_class_scatter(Xpca, Y)
  Sb <- between_class_scatter(Xpca, Y, colMeans(Xpca))
  
  gmeans <- group_means(Y, Xpca)
  
  E_i <- RSpectra::eigs_sym(Sw, k=di)
  proj_di <- E_i$vectors %*% diag(1/sqrt(E_i$values))
  
  gmeans_proj <- (gmeans %*% proj_di)
  
  E_l <- pca(gmeans_proj, ncomp=dl)
  
  proj_dl <- E_l$v
  
  proj_final <- proj_dp %*% proj_di %*% proj_dl
  s <- Xp %*% proj_final
  ret <- multivarious::discriminant_projector(v=proj_final, s=s, 
                                       sdev=apply(s,2,sd),
                                       dp=dp,
                                       dl=dl,
                                       di=di,
                                       labels=Y,classes="pca_lda")
}









