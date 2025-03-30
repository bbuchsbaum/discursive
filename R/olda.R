#' Orthogonal Linear Discriminant Analysis
#'
#' This function performs Orthogonal Linear Discriminant Analysis (OLDA) on a given dataset.
#' OLDA finds orthogonal linear discriminants that maximize the separation between classes. 
#' It is useful for dimensionality reduction and classification tasks.
#'
#' @param X A numeric matrix where columns represent features and rows represent samples.
#' @param Y A factor or numeric vector representing class labels for each sample.
#' @param preproc A preprocessing function (from \code{multivarious}) to apply to the data. 
#'   Default is \code{pass()} (no preprocessing).
#' @return A \code{discriminant_projector} object containing:
#'   \itemize{
#'     \item \code{v}    : The orthogonal loadings (projection matrix).
#'     \item \code{s}    : The projected scores.
#'     \item \code{sdev} : Standard deviations of the score dimensions.
#'     \item \code{labels}: The class labels.
#'     \item \code{preproc}: The preprocessing object.
#'   }
#' @export
#' @examples
#' \dontrun{
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' Y <- iris[, 5]
#' res <- olda(X, Y)
#' print(res)
#' }
#'
#' @details
#' The function proceeds through the following steps:
#' 1. **ULDA Projection**: The data is first projected using Uncorrelated Linear Discriminant Analysis (ULDA).
#' 2. **QR Decomposition**: The projection matrix from ULDA is then orthogonalized using QR decomposition.
#' 3. **Final Projection**: The orthogonalized projection matrix is applied to the (preprocessed) data,
#'    and a \code{discriminant_projector} object is returned.
#'
#' @seealso \code{\link{ulda}}, \code{\link{qr}}, \code{\link{qr.Q}}, \code{\link[multivarious]{discriminant_projector}}
olda <- function(X, Y, preproc = pass()) {
  # 1) Run ULDA
  res_lda <- ulda(X, Y, preproc = preproc)
  
  # 2) Extract the original loadings (p x r)
  v_init <- coef(res_lda)  # e.g., columns are ULDA directions
  
  # 3) Orthogonalize using QR
  q_decomp <- qr(v_init)
  v_ortho  <- qr.Q(q_decomp)  # orthonormal basis
  
  # 4) Re-project the data with the orthonormal loadings 
  #    so we can store the training scores in the projector.
  #    We'll get the preproc from res_lda$preproc (assuming 'ulda' returns it).
  
  # a) Retrieve the final preprocessor (may be pass() or something else).
  preproc_final <- res_lda$preproc
  # b) Transform X with the same pipeline
  Xp <- init_transform(preproc_final, X)
  # c) Scores = Xp %*% v_ortho
  s_mat <- Xp %*% v_ortho
  
  # 5) Build the discriminant_projector
  #    Provide labels (Y) or possibly res_lda$labels if 'ulda' stores them
  #    sdev = standard deviation of each dimension
  dp <- multivarious::discriminant_projector(
    v       = v_ortho,
    s       = s_mat,
    sdev    = apply(s_mat, 2, sd),
    preproc = preproc_final,
    labels  = factor(Y),       # or factor(res_lda$labels) if that is stored
    classes = "olda"
  )
  
  return(dp)
}
