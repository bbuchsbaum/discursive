% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gmmsd.R
\name{gmmsd}
\alias{gmmsd}
\title{Generalized Multiple Maximum Scatter Difference (GMMSD)}
\usage{
gmmsd(X, y, c = 1, dim = 2, preproc = multivarious::center())
}
\arguments{
\item{X}{A numeric matrix (n x d), where n is the number of samples (rows) and d
is the number of features (columns).}

\item{y}{A factor or numeric vector of length n representing class labels for each sample.
If numeric, it will be internally converted to a factor.}

\item{c}{A numeric balance parameter scaling the within-class scatter matrix.
Typically a positive value. Default is 1.}

\item{dim}{The number of dimensions (features) to retain in the transformed feature space.}

\item{preproc}{A \code{pre_processor} object from \pkg{multivarious}
(e.g. \code{center()}, \code{scale()}). Defaults to \code{center()}.}
}
\value{
A \code{discriminant_projector} object (subclass can be \code{"gmmsd"}) containing:
\itemize{
\item \code{v}    : A \code{d x dim} loading/projection matrix.
\item \code{s}    : An \code{n x dim} score matrix (the data projected onto the new axes).
\item \code{sdev} : Standard deviations of each dimension in \code{s}.
\item \code{labels}: The class labels.
\item \code{preproc}: The preprocessing object used.
}
}
\description{
This function implements the GMMSD method for feature extraction. It solves a
symmetric generalized eigenvalue problem to find a projection that maximizes the
difference between the between-class scatter and a scaled within-class scatter.
The method uses a QR decomposition to enhance computational efficiency, making it
suitable for high-dimensional data. The preprocessing object must come from the
\pkg{multivarious} package.
}
\examples{
\dontrun{
library(multivarious)

data(iris)
X <- as.matrix(iris[, -5])
y <- iris$Species

# By default, this will center the data prior to GMMSD
dp <- gmmsd(X, y, c = 1, dim = 2)

# Inspect the projector
print(dp)

# Project the original data
scores <- dp$s
# or equivalently, project(dp, X)
}
}
