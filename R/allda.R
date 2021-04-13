## https://paperpile.com/app/p/5541796c-f768-083e-ba28-94fd779ef7e8
## Nie, Feiping, Zheng Wang, Rong Wang, Zhen Wang, and Xuelong Li. 2020. 
## “Adaptive Local Linear Discriminant Analysis.” 
## ACM Transactions on Knowledge Discovery from Data, 9, 14 (1): 1–19.
#' @examples 
#' 
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' Y <- iris[,4]
#' 
allda <- function(X, Y, preproc=center(), dp=min(dim(X)), ddi=dp-1, dl=length(unique(Y))-1, r=2, k=5) {
  
  ## compute class graph, S, within class weights are 1/k, otherwise 0
  ## A <- S^r
  ## L_A <- D - (A +t(A))/2
  cg <- neighborweights::class_graph(Y)
  
 
  nn <- neighborweights:::find_nn_among.class_graph(cg, xx, k=k)
  A <- neighborweights:::dist_to_sim.Matrix(neighborweights:::adjacency.nnsearch(nn), method="binary")
  A <- A^r
  A <- (A + t(A))/2
 
  
  D <- Matrix::Diagonal(x=Matrix::rowSums(A))
  LA <- D - A
  
  procres <- multivarious::prep(preproc)
  Xp <- init_transform(procres, X)
  
  pca_basis <- multivarious::pca(Xp, ncomp=dp)
  proj_dp <- pca_basis$v
  Xpca <- scores(pca_basis)
  
  St <- total_scatter(Xpca, colMeans(Xpca))
  #St_inv <- solve(St)
  
  #M <- St_inv %*% (t(Xpca) %*% LA %*% Xpca)
  M <- solve(St, (t(Xpca) %*% LA %*% Xpca))
  W <- RSpectra::eigs(M, k=nrow(M))$vectors
  
  scores <- Xp %*% W
  nn <- neighborweights:::find_nn_among.class_graph(cg, scores, k=k)
  A <- neighborweights:::adjacency.nnsearch(nn)
  diag(A) <- 0
  
  ## W <- eigenvectors of S_t_inv %*% X %*% L_A %*% t(X)
  ## update S, equation 25, e..g compute S using k-nearest neighbors in discriminant space
  
  
}