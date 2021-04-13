

#' @inheritParam ulda
#' @export
olda <- function(X, Y, preproc=pass()) {
  res <- ulda(X,Y, preproc)
  v <- components(res)
  q <- qr(v)
  v <- qr.Q(q)
  
  multivarious::projector(v, res$preproc, classes="olda")
}


#
#utrda <- function(X, Y, preproc=pass()) {
#  
#}