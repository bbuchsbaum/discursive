% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fast_olda.R
\name{fastolda}
\alias{fastolda}
\title{Fast Orthogonal LDA}
\usage{
fastolda(X, Y, preproc = center(), reg = 0.01)
}
\arguments{
\item{X}{A \code{matrix} (n x d) with n samples and d features.}

\item{Y}{A \code{factor} with length n, providing the class/group label for each sample.}

\item{preproc}{A pre-processing step, such as \code{center()}, from \code{multivarious}. Default is \code{center()}.}

\item{reg}{A \code{numeric} regularization parameter (default = 0.01). This is used to ensure invertibility of certain matrices.}
}
\value{
A \code{discriminant_projector} object containing:
\itemize{
\item \code{rotation}: The matrix of loadings (d x r) where r is the reduced dimension.
\item \code{s}: The scores matrix (n x r), i.e., \code{X \%*\% rotation}.
\item \code{sdev}: Standard deviations of the scores.
\item \code{labels}: The class labels.
\item \code{preproc}: The preprocessing object.
}
}
\description{
Perform a fast Orthogonal Linear Discriminant Analysis (OLDA) based on provided data and class labels.
}
\details{
This function performs OLDA by pre-processing the data, computing difference-based scatter matrices, and then solving for a discriminant projection.
The final result is returned as a \code{discriminant_projector} object from the \code{multivarious} package.
}
\examples{
data(iris)
X <- as.matrix(iris[, 1:4])
Y <- iris[, 5]
res <- fastolda(X, Y)
}
