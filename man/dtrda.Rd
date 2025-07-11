% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dtr.R
\name{dtrda}
\alias{dtrda}
\title{Discriminant analysis with Trace Regularization (DTR)}
\usage{
dtrda(X, Y, preproc = multivarious::center(), d = 2, alpha)
}
\arguments{
\item{X}{numeric matrix of predictors, of dimension n x p.}

\item{Y}{factor variable of class labels, of length n.}

\item{preproc}{A preprocessing function to apply to the data. Default is centering.}

\item{d}{integer, the dimension of the discriminant subspace. Must be <= K-1 where K is the number of classes.}

\item{alpha}{numeric in \link{0,1} controlling the trade-off between between-class and within-class scatters.}
}
\value{
An S3 object of class "discriminant_projector" containing the transformation matrix W,
the transformed scores, and related metadata.
}
\description{
Finds a low-dimensional discriminant subspace that maximizes
the between-class scatter while controlling the within-class scatter.
}
\examples{
X = matrix(rnorm(100*1000), 100, 1000) 
y = sample(1:3, 100, replace=TRUE)
V = dtrda(X, y, d=2, alpha=0.5)
Xp = X \%*\% V  # project data onto discriminant subspace

}
\references{
Ahn, J., Chung, H. C., & Jeon, Y. (2021). Trace Ratio Optimization for High-Dimensional Multi-Class Discrimination. Journal of Computational and Graphical Statistics, 30(1), 192-203. \doi{10.1080/10618600.2020.1807352}
}
