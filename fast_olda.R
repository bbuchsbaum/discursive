


#' group_means
#' 
#' @param Y \code{factor} variable defining the groups
#' @param X \code{matrix} defining the matrix data to be group-wise averaged
#' @export
group_means <- function(Y, X) {
  
  if (all(table(Y) == 1)) {
    row.names(X) <- names(table(Y))
    X
  } else {
    Rs <- rowsum(X,Y)
    yt <- table(Y)
    ret <- sweep(Rs, 1, yt, "/")
    row.names(ret) <- names(yt)
    ret
  }
}

#' @inheritParams ulda
#' @param reg regularization parameter
#' @export
fastolda <- function(X, Y, preproc=center(), reg=.01) {
  
  procres <- preproc %>% prep()
  X <- init_transform(procres)
  
  freq <- table(Y)
  
  Ht_diff <- compute_Htdiff(X)
  Ht_diffS <- crossprod(Ht_diff)
  
  m <- rowSums(Ht_diff)/nrow(X)
  HT <- sweep(Ht_diffS, 1, crossprod(m, Ht_diff), "-")
  HT <- cbind(t(-m %*% Ht_diff), HT)
  
  ## need to compute HT from Ht_diffS ...
  ##HT <- crossprod(Ht_diff, Ht)
  
  ST <- tcrossprod(HT, HT)
  
  RT <- chol(ST)
  RT_inv <- chol2inv(RT)
  
  #HB <- crossprod(Ht_diff, Hb)
  
  HBs <- t(rowsum(t(HT), Y))
  HB <- sweep(HBs, 2, freq * sqrt(freq), "/")
  
  Hfinal <- Ht_diff %*% RT_inv %*% HB[, 1:(length(freq)-1)]
  
  GR <- qr(Hfinal)
  v <- qr.Q(GR)
  
  s <- X %*% v
  
  multivarious::discriminant_projector(v, s=s,
                                       sdev=apply(proj_final,2,sd),
                                       labels=Y,
                                       preproc=procres, classes="fast_olda")
  
}