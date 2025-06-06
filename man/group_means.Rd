% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fast_olda.R
\name{group_means}
\alias{group_means}
\title{Compute Group Means}
\usage{
group_means(Y, X)
}
\arguments{
\item{Y}{A \code{factor} indicating group membership for each row of \code{X}.}

\item{X}{A \code{matrix} (n x d), where n is the number of samples and d is the number of features.}
}
\value{
A matrix of group means, where the number of rows equals the number of groups, and each row is the mean vector for that group.
}
\description{
Compute the group-wise mean vectors for a given data matrix.
}
\details{
Given a data matrix \code{X} and a factor \code{Y}, this function computes the mean of the rows of \code{X} for each level of \code{Y}.
}
