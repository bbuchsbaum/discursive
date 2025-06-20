#' discursive: Graph-Based Discriminant Analysis
#'
#' Implements discriminant analysis algorithms including DGPAGE and
#' several orthogonal LDA variants. Provides utilities for building
#' similarity and diversity graphs as well as Rcpp-based scatter
#' computations.
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{dgpage_discriminant}}: fit the DGPAGE model.
#'   \item \code{\link{dgpage_predict}}: predict new observations.
#' }
#'
#' @docType package
#' @name discursive
#' @aliases discursive-package
"_PACKAGE"

## usethis namespace: start
#' @useDynLib discursive, .registration = TRUE
## usethis namespace: end
NULL
