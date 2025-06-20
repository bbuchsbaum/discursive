#' Generalized Multiple Maximum Scatter Difference (GMMSD)
#'
#' This function implements the GMMSD method for feature extraction. It solves a
#' symmetric generalized eigenvalue problem to find a projection that maximizes
#' the difference between the between-class scatter and a scaled within-class
#' scatter. The method uses a QR decomposition to enhance computational
#' efficiency, making it suitable for high-dimensional data. The preprocessing
#' object must come from the \pkg{multivarious} package.
#'
#' @param X A numeric matrix (n x d), where n is the number of samples (rows) and d
#'          is the number of features (columns).
#' @param y A factor or numeric vector of length n representing class labels for each sample.
#'          If numeric, it will be internally converted to a factor.
#' @param c A numeric balance parameter scaling the within-class scatter matrix.
#'          Typically a positive value. Default is 1.
#' @param dim The number of dimensions (features) to retain in the transformed feature space.
#' @param preproc A \code{pre_processor} object from \pkg{multivarious}
#'   (e.g. \code{center()}, \code{scale()}). Defaults to \code{center()}.
#'
#' @return A \code{discriminant_projector} object (subclass can be \code{"gmmsd"}) containing:
#' \itemize{
#'   \item \code{v}    : A \code{d x dim} loading/projection matrix.
#'   \item \code{s}    : An \code{n x dim} score matrix (the data projected onto the new axes).
#'   \item \code{sdev} : Standard deviations of each dimension in \code{s}.
#'   \item \code{labels}: The class labels.
#'   \item \code{preproc}: The preprocessing object used.
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(multivarious)
#'
#' data(iris)
#' X <- as.matrix(iris[, -5])
#' y <- iris$Species
#'
#' # By default, this will center the data prior to GMMSD
#' dp <- gmmsd(X, y, c = 1, dim = 2)
#'
#' # Inspect the projector
#' print(dp)
#'
#' # Project the original data
#' scores <- dp$s
#' # or equivalently, project(dp, X)
#' }
gmmsd <- function(X, y, c = 1, dim = 2, preproc = multivarious::center()) {
  
  # 1) Convert y to factor if necessary
  if (!is.factor(y)) {
    y <- factor(y)
  }
  
  # 2) Preprocessing step: center/scale/etc. if desired
  procres <- multivarious::prep(preproc)
  Xp <- multivarious::init_transform(procres, X)  # Xp is the preprocessed data (n x d)
  
  # 3) Mean-center check: Xp might already be centered by default if preproc=center().
  #    If your between_class_scatter() / within_class_scatter() do not handle
  #    centering internally, then pass Xp as is:
  
  # 4) Use QR decomposition on t(Xp). If n >= d, qr.Q(...) yields a (d x d)
  #    orthonormal basis.
  qr.decomp <- qr(t(Xp))
  Q1 <- qr.Q(qr.decomp)  # d x d
  
  # 5) Compute between-class and within-class scatter on Xp
  Sb <- between_class_scatter(Xp, y)
  Sw <- within_class_scatter(Xp, y)
  
  # 6) Form M = Q1^T (Sb - c * Sw) Q1
  M <- t(Q1) %*% (Sb - c * Sw) %*% Q1
  
  # 7) Solve eigenvalue problem (M is symmetric by construction)
  eigres <- eigen(M, symmetric = TRUE)
  if (dim > ncol(Q1)) {
    stop("dim exceeds available dimensions")
  }
  # By default eigen() returns eigenvalues in decreasing order.
  # We'll take the top 'dim' eigenvectors
  W <- Q1 %*% eigres$vectors[, 1:dim, drop = FALSE]  # shape (d x dim)
  
  # 8) Project the data: Xp %*% W gives an (n x dim) matrix of scores
  s <- Xp %*% W
  
  # 9) Build the discriminant_projector
  #    v = W (d x dim)
  #    s = the n x dim scores
  #    sdev = std dev of each dimension in s
  dp <- multivarious::discriminant_projector(
    v       = W,
    s       = s,
    sdev    = apply(s, 2, sd),
    preproc = procres,
    labels  = y,
    classes = "gmmsd"
  )
  
  return(dp)
}

