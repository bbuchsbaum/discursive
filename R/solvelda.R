solvelda <- function(Sw, Sb) {
  M <- solve(Sw, Sb)
  ret <- RSpectra::eigs(M, k=nrow(M))
  projector(v=ret$vectors, preproc=prep(pass()), classes="eigenlda")
}