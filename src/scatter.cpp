#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix withinSScpp(NumericMatrix X, IntegerVector group) {
  int n = X.nrow(), p = X.ncol();
  NumericMatrix Within(p, p);
  int g = max(group);
  NumericMatrix means(g, p);  // Matrix for group means
  std::vector<int> counts(g, 0);
  
  // Calculate means and counts for each group
  for (int i = 0; i < n; ++i) {
    int grp = group[i] - 1;  // zero-based index
    for (int j = 0; j < p; ++j) {
      means(grp, j) += X(i, j);
    }
    counts[grp]++;
  }
  for (int grp = 0; grp < g; ++grp) {
    for (int j = 0; j < p; ++j) {
      means(grp, j) /= counts[grp];
    }
  }
  
  // Accumulate the within-class scatter matrix
  for (int i = 0; i < n; ++i) {
    int grp = group[i] - 1;
    NumericVector centered(p);
    for (int j = 0; j < p; ++j) {
      centered[j] = X(i, j) - means(grp, j);
    }
    
    
    for (int k = 0; k < p; ++k) {
      for (int j = 0; j < p; ++j) {
        Within(k, j) += centered[k] * centered[j];
      }
    }
  }
  
  return Within;
}

// [[Rcpp::export]]
NumericMatrix betweenSScpp(NumericMatrix X, IntegerVector group) {
  int n = X.nrow(), p = X.ncol();
  NumericMatrix Between(p, p);
  int g = max(group);
  NumericMatrix means(g, p);  // Matrix for group means
  NumericVector overall_mean(p);
  std::vector<int> counts(g, 0);
  
  // Calculate overall mean and group means
  for (int i = 0; i < n; ++i) {
    overall_mean += X.row(i);
    int grp = group[i] - 1; // zero-based index for group
    for (int j = 0; j < p; ++j) {
      means(grp, j) += X(i, j);
    }
    counts[grp]++;
  }
  overall_mean = overall_mean / n;
  
  for (int grp = 0; grp < g; ++grp) {
    for (int j = 0; j < p; ++j) {
      means(grp, j) /= counts[grp];
    }
  }
  

  // Calculate between-class scatter matrix
  for (int grp = 0; grp < g; ++grp) {
    NumericVector mean_diff(p);
    for (int j = 0; j < p; ++j) {
      mean_diff[j] = means(grp, j) - overall_mean[j];
    }
    

    for (int i = 0; i < p; ++i) {
      for (int j = 0; j < p; ++j) {
        Between(i, j) += counts[grp] * mean_diff[i] * mean_diff[j];
      }
    }
  }
  
  return Between;
}
